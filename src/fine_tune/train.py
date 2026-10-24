import torch

import numpy as np

import time
from tqdm.auto import tqdm

import json
from copy import deepcopy
from collections import deque

from src.config.types import TrainingObjects, TrainingResults, EpochHistory, ModelHistory
from src.config.paths import MODEL_DIR

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

        mini_batches = len(dataloader)
        num_batches = mini_batches / accum_steps

        history = EpochHistory()
        optimizer.zero_grad()
        for i, (imgs, coords) in enumerate(dataloader):
            imgs = imgs.to(device)

            batch_start = (i // accum_steps) * accum_steps  # gets lowest starting value 
            batch_size = min(accum_steps, num_batches - batch_start)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(imgs) #TODO find out what YOLO returns
                loss, metrics = criterion(outputs, coords)

            scaler.scale(loss / batch_size).backward()

            # gradient accumulation, lets it simulate larger batches
            if (i + 1) % batch_size == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            history.append(loss.item(), metrics)

        if num_batches == 0:
            raise RuntimeError("Training dataloader produced no batches")
        
        if num_batches % accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
        return history

    @torch.inference_mode()
    def evaluate(self, dataloader):
        model = self.objs.model
        criterion = self.objs.criterion
        device = self.objs.device

        model.eval()

        history = EpochHistory()
        for imgs, coords in dataloader:
            imgs = imgs.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(imgs) # TODO here too
                loss, metrics = criterion(outputs, coords)

            history.append(loss.item(), metrics)
               
        num_batches = len(dataloader)
        if num_batches == 0:
            raise RuntimeError("Evaluation dataloader produced no batches")
        return history

    def train_model(
            self, 
            epochs: int,
            train_loader,
            val_loader
        ):
        if epochs == 0:
            raise ValueError("Cannot train model for 0 epochs")

        best_eval_loss = float("inf")
        best_epoch = None
        best_state = None

        # Window of recent losses to see if there is none/minimal change
        eval_window = deque(maxlen=5)

        pbar = tqdm(total=epochs, desc="Training")

        history = ModelHistory()
        try:
            for epoch in range(epochs):
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                train_history = self.train_one_epoch(train_loader)
                train_time = time.perf_counter() - start
                print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")

                start = time.perf_counter()
                eval_history = self.evaluate(val_loader)
                eval_time = time.perf_counter() - start

                eval_loss = eval_history.avg_loss
                train_loss = train_history.avg_loss

                history.append(train_history, eval_history)
                
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_epoch = epoch + 1

                    # TODO may need to be changed as it keeps the model on VRAM will verify
                    best_state = deepcopy(self.objs.model.state_dict())

                tqdm.write(
                    f"Epoch {epoch + 1:02d} | "
                    f"Train loss {train_loss:.4f} | "
                    f"Eval loss {eval_loss:.4f} | "
                    f"Gap {abs(eval_loss - train_loss):.4f} | "
                    f"Train time {train_time} | "
                    f"Eval time {eval_time} | "
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
                        break

            if best_state is None or best_epoch is None:
                raise RuntimeError("Training completed without a valid validation result")

        except Exception as e:
            return TrainingResults(
                history=history,
                best_state=best_state,
                best_eval_loss=best_eval_loss,
                best_epoch=best_epoch,
                error=e
            )
        finally:
            pbar.close()

        return TrainingResults(
            history=history,
            best_state=best_state,
            best_eval_loss=best_eval_loss,
            best_epoch=best_epoch,
            )

    @staticmethod
    def save_results(results, model_info, timestamp):
        EXPER_DIR = MODEL_DIR / timestamp
        EXPER_DIR.mkdir(parents=True, exist_ok=True)

        with open(EXPER_DIR / "metadata.json", "w") as f:
            json.dump(model_info, f, indent=4)

        # TODO convert results to csv/json

        if results.best_state is not None:
            cpu_state = {
                key: value.detach().cpu()
                for key, value in results.best_state.items()
            }
            torch.save(cpu_state, EXPER_DIR / "model.pt")
            print(f"Best Model Saved (val_loss={results.best_eval_loss:.4f}) at epoch {results.best_epoch}")

