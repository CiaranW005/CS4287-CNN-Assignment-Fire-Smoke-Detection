"""
    python -m scripts.training.train
"""
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

import json
import pandas as pd
from datetime import datetime

import subprocess
import gc

from src.fine_tune import Trainer, HybridModel
from src.fine_tune.losses import YOLOLoss
from src.data.preprocessing import ModelData, train_transform, eval_transform

from src.utils import get_device, parse_boxes, collate_fn
from src.config.paths import CSV_DIR, MODEL_DIR, DATASET_DIR
from src.config.types import ModelParameters, ModelInfo, TrainingObjects

if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_DIR / "data/metadata.csv")
    df["boxes"] = df["boxes"].apply(lambda value: parse_boxes(json.loads(value)))

    device = get_device()
    git_commit = subprocess.check_output( ["git", "rev-parse", "HEAD"], text=True ).strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    NOTES = [
        "Test"
    ]

    PARAMS: ModelParameters = {
        "seed": 42,
        "num_workers": 8,
        "pin_memory": device.type == "cuda",
        
        "batch_size": 14,
        "epochs": 50,
        "accumulation_steps": 2,

        "learning_rate": 1e-3,
        "weight_decay": 1e-4,

        "model_name": "model.pt",
    }

    MODEL_INFO: ModelInfo = {
        "timestamp": timestamp,
        "notes" : NOTES,
        "git_commit" : git_commit,
        "parameters": PARAMS,
    }

    train_data = list(
        df.loc[df["split"] == "train", ["path", "boxes"]]
        .itertuples(index=False, name=None)
    )

    val_data = list(
        df.loc[df["split"] == "val", ["path", "boxes"]]
        .itertuples(index=False, name=None)
    )

    test_data = list(
        df.loc[df["split"] == "test", ["path", "boxes"]]
        .itertuples(index=False, name=None)
    )

    train_dataset = ModelData(DATASET_DIR, train_data, train_transform)
    val_dataset = ModelData(DATASET_DIR, val_data, eval_transform)
    test_dataset = ModelData(DATASET_DIR, test_data, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=PARAMS["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=PARAMS["num_workers"],
        pin_memory=PARAMS["pin_memory"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=PARAMS["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=PARAMS["num_workers"],
        pin_memory=PARAMS["pin_memory"]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=PARAMS["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=PARAMS["num_workers"],
        pin_memory=PARAMS["pin_memory"]
    )

    model = HybridModel().to(device)

    optimiser = AdamW(
        [
            #pretrained backbone: smaller steps so the ImageNet features aren't wiped out
            {"params": model.backbone.parameters(), "lr": PARAMS["learning_rate"] * 0.1},
            # adapter, neck and head start from random init so use the full rate
            {"params": [p for name, p in model.named_parameters() if not name.startswith("backbone.")]},
        ],
        lr=PARAMS["learning_rate"], 
        weight_decay=PARAMS["weight_decay"]
    )

    criterion = YOLOLoss(model) 

    scaler = torch.amp.GradScaler("cuda")

    training_objs = TrainingObjects(
        model=model,
        optimizer=optimiser,
        criterion=criterion,
        scaler=scaler,
        device=device,
        accumulation_steps=PARAMS["accumulation_steps"],
    )

    trainer = Trainer(objs=training_objs)

    results = trainer.train_model(
        epochs=PARAMS["epochs"],
        train_loader=train_loader,
        val_loader=val_loader
    )
    if results.error is not None:
        raise results.error
    
    try:        
        if results.best_state is not None:
            training_objs.model.load_state_dict(results.best_state)
            test_history = trainer.evaluate(test_loader)
            print(
                f"Test mAP50 {test_history.map50:.3f} | "
                f"mAP50-95 {test_history.map50_95:.3f} | "
                f"per class [smoke, fire] {test_history.map_per_class}"
            )

            trainer.save_results(
                results,
                model_info=MODEL_INFO,
                timestamp=timestamp
            )
    finally:
        del trainer
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()
