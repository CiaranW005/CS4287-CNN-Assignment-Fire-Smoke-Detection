import torch

import numpy as np
import pandas as pd

import time
from tqdm.auto import tqdm

import json
from copy import deepcopy
from collections import deque

from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_convert
from ultralytics.utils.nms import non_max_suppression

from src.config.types import TrainingObjects, TrainingResults, EpochHistory, ModelHistory, EvaluationMetrics
from src.config.paths import MODEL_DIR

from src.utils.get_mean import mean_valid

class Trainer:
    def __init__(self, objs: TrainingObjects):
        self.objs = objs

    def train_one_epoch(self, dataloader):
        model = self.objs.model
        criterion = self.objs.criterion
        optimizer = self.objs.optimizer
        device = self.objs.device
        accum_steps = self.objs.accumulation_steps
        scaler = self.objs.scaler

        model.train()

        num_batches = len(dataloader)
        if num_batches == 0:
            raise RuntimeError("Training dataloader produced no batches")

        history = EpochHistory()
        optimizer.zero_grad()
        for i, (imgs, coords) in enumerate(dataloader):
            imgs = imgs.to(device)

            batch_start = (i // accum_steps) * accum_steps  # gets lowest starting value 
            batch_size = min(accum_steps, num_batches - batch_start)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(imgs)
                loss, metrics = criterion(outputs, coords)

            scaler.scale(loss / batch_size).backward()

            # gradient accumulation, lets it simulate larger batches
            if (i + 1) % accum_steps == 0 or (i + 1) == num_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            history.append(loss.item(), metrics)
        
        return history

    @torch.inference_mode()
    def evaluate(self, dataloader):
        model = self.objs.model
        criterion = self.objs.criterion
        device = self.objs.device

        if len(dataloader) == 0:
            raise RuntimeError("Evaluation dataloader produced no batches")

        model.eval()

        # COCO-style mAP over the whole split, class_metrics gives fire and smoke separately
        map_metrics = MeanAveragePrecision(
            box_format="xyxy", iou_type="bbox", 
            class_metrics=True,  extended_summary=True,
            max_detection_thresholds=[1, 10, 100]
        )

        history = EpochHistory()
        for imgs, targets in dataloader:
            imgs = imgs.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(imgs) # eval mode: (decoded boxes, raw predictions)
                loss, metrics = criterion(outputs, targets)

            history.append(loss.item(), metrics)

            # same thresholds Ultralytics validates with, low conf keeps every recall level
            detections = non_max_suppression(outputs, conf_thres=0.001, iou_thres=0.7, max_det=300)

            predictions = [
                {
                    "boxes": det[:, :4].float().cpu(),
                    "scores": det[:, 4].float().cpu(),
                    "labels": det[:, 5].long().cpu(),
                }
                for det in detections
            ]
            map_metrics.update(predictions, self._ground_truth(targets, len(imgs), imgs.shape[-2:]))

        results = map_metrics.compute()
        recall = results["recall"]

        history.evaluation = EvaluationMetrics(
            map50 = results["map_50"].item(),
            map50_95 = results["map"].item(),
            map_per_class = results["map_per_class"].tolist(),

            map_small = results["map_small"].item(),
            map_medium = results["map_medium"].item(),
            map_large = results["map_large"].item(),

            mar_100 = results["mar_100"].item(),
            mar_small = results["mar_small"].item(),
            mar_medium = results["mar_medium"].item(),
            mar_large = results["mar_large"].item(),

            recall50 = mean_valid(recall[0, :, 0, 2]),
            recall50_small = mean_valid(recall[0, :, 1, 2]),
            recall50_medium = mean_valid(recall[0, :, 2, 2]),
            recall50_large = mean_valid(recall[0, :, 3, 2]),

            recall75 = mean_valid(recall[5, :, 0, 2]),
            recall75_small = mean_valid(recall[5, :, 1, 2]),
            recall75_medium = mean_valid(recall[5, :, 2, 2]),
            recall75_large = mean_valid(recall[5, :, 3, 2]),
        )
      
        return history

    @staticmethod
    def _ground_truth(targets, batch_size, image_size):
        """Converts collated targets to per-image pixel xyxy boxes for MeanAveragePrecision

        targets["bboxes"] are normalised x_center, y_center, width, height
        predictions from NMS are pixel x1, y1, x2, y2 so the targets are scaled to match
        """
        height, width = image_size
        scale = torch.tensor([width, height, width, height])
        boxes = box_convert(targets["bboxes"] * scale, in_fmt="cxcywh", out_fmt="xyxy")

        return [
            {
                "boxes": boxes[targets["batch_idx"] == i],
                "labels": targets["cls"][targets["batch_idx"] == i].long(),
            }
            for i in range(batch_size)
        ]

    def train_model(
            self, 
            epochs: int,
            train_loader,
            val_loader
        ):
        if epochs == 0:
            raise ValueError("Cannot train model for 0 epochs")

        best_map50_95 = float("-inf")
        best_epoch = None
        best_state = None

        # Window of recent losses to see if there is none/minimal change
        eval_window = deque(maxlen=5)
        decreased_lr = False

        pbar = tqdm(total=epochs, desc="Training")

        history = ModelHistory()
        try:
            for epoch in range(epochs):
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                train_history = self.train_one_epoch(train_loader)
                train_time = time.perf_counter() - start
                vram = f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB"

                start = time.perf_counter()
                eval_history = self.evaluate(val_loader)
                eval_time = time.perf_counter() - start

                eval_loss = eval_history.avg_loss
                train_loss = train_history.avg_loss

                history.append(train_history, eval_history)
                
                if eval_history.evaluation.map50_95 > best_map50_95:
                    best_map50_95 = eval_history.evaluation.map50_95
                    best_epoch = epoch + 1

                    # TODO may need to be changed as it keeps the model on VRAM will verify
                    best_state = deepcopy(self.objs.model.state_dict())

                tqdm.write(
                    f"Epoch {epoch + 1:02d} | "
                    f"Train loss {train_loss:.4f} | "
                    f"Eval loss {eval_loss:.4f} | "
                    f"Gap {abs(eval_loss - train_loss):.4f} | "
                    f"mAP50 {eval_history.evaluation.map50:.3f} | "
                    f"mAP50-95 {eval_history.evaluation.map50_95:.3f} | "
                    f"Train time {train_time:.2f} | "
                    f"Eval time {eval_time:.2f} | "
                    f"VRAM usage {vram} | " 
                )

                pbar.update(1)

                eval_window.append(eval_loss)
                # Early stopping mechanism
                if len(eval_window) == eval_window.maxlen:
                    window = np.asarray(eval_window)

                    x = np.arange(len(window))
                    # y = mx + c where y = window
                    slope = np.polyfit(x, window, 1)[0] # polyfit returns [m, c]

                    relative_slope = slope / (abs(window.mean()) + 1e-8)
                    # if it doesnt improve by more than -0.1% stop training
                    if relative_slope > -1e-3:
                        if not decreased_lr:
                            for group in self.objs.optimizer.param_groups:
                                group["lr"] *= 0.5

                            decreased_lr = True
                            eval_window.clear()

                            tqdm.write(
                                f"Validation loss plateau detected. Reducing LR to {self.objs.optimizer.param_groups[0]['lr']:.2e}"
                            )
                            continue
                        break

            if best_state is None or best_epoch is None:
                raise RuntimeError("Training completed without a valid validation result")

        except Exception as e:
            return TrainingResults(
                history=history,
                best_state=best_state,
                best_map50_95=best_map50_95,
                best_epoch=best_epoch,
                error=e
            )
        finally:
            pbar.close()

        return TrainingResults(
            history=history,
            best_state=best_state,
            best_map50_95=best_map50_95,
            best_epoch=best_epoch,
            )

    @staticmethod
    def save_results(results, model_info, timestamp):
        EXPER_DIR = MODEL_DIR / timestamp
        EXPER_DIR.mkdir(parents=True, exist_ok=True)

        with open(EXPER_DIR / "metadata.json", "w") as f:
            json.dump(model_info, f, indent=4)

        rows = [
            {
            "epoch": epoch,
            "train_loss": train.avg_loss,
            "val_loss": val.avg_loss,
            
            "map50": val.evaluation.map50,
            "map50_95": val.evaluation.map50_95,
            "map_small": val.evaluation.map_small,
            "map_medium": val.evaluation.map_medium,
            "map_large": val.evalutation.map_large,

            "mar_100": val.evaluation.mar_100,
            "mar_small": val.evaluation.mar_small,
            "mar_medium": val.evaluation.mar_medium,
            "mar_large": val.evalutation.mar_large,

            "recall50": val.evaluation.recall50,
            "recall50_small": val.evaluation.recall50_small,
            "recall50_medium": val.evaluation.recall50_medium,
            "recall50_large": val.evaluation.recall50_large,

            "recall75": val.evaluation.recall75,
            "recall75_small": val.evaluation.recall75_small,
            "recall75_medium": val.evaluation.recall75_medium,
            "recall75_large": val.evaluation.recall75_large,
            }
            for epoch, (train, val) in enumerate(zip(results.history.train, results.history.eval), start=1)
        ]
        pd.DataFrame(rows).to_csv(EXPER_DIR / "history.csv", index=False)

        if results.best_state is not None:
            cpu_state = {
                key: value.detach().cpu()
                for key, value in results.best_state.items()
            }
            torch.save(cpu_state, EXPER_DIR / "model.pt")
            print(f"Best Model Saved (map50_95={results.best_map50_95:.4f}) at epoch {results.best_epoch}")

