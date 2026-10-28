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

from src.utils import get_device, parse_boxes, collate_fn, print_evaluation
from src.config.paths import CSV_DIR, MODEL_DIR, DATASET_DIR
from src.config.types import ModelParameters, ModelInfo, TrainingObjects

if __name__ == "__main__":
    model_dir = MODEL_DIR / "20261028_114248"
    print(model_dir)

    df = pd.read_csv(CSV_DIR / "data/metadata.csv")
    df["boxes"] = df["boxes"].apply(lambda value: parse_boxes(json.loads(value)))

    device = get_device()
    git_commit = subprocess.check_output( ["git", "rev-parse", "HEAD"], text=True ).strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(model_dir / "metadata.json", "r") as f:
        meta = json.load(f)

    PARAMS = meta["parameters"]

    test_data = list(
        df.loc[df["split"] == "test", ["path", "boxes"]]
        .itertuples(index=False, name=None)
    )
    test_dataset = ModelData(DATASET_DIR, test_data, eval_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=PARAMS["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=PARAMS["num_workers"],
        pin_memory=PARAMS["pin_memory"]
    )

    model = HybridModel().to(device)
    state_dict = torch.load(
        model_dir / "model.pt",
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    criterion = YOLOLoss(model) 

    training_objs = TrainingObjects(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters()),
        criterion=criterion,
        scaler=torch.GradScaler(),
        device=device,
        accumulation_steps=0,
    )

    trainer = Trainer(objs=training_objs)

    try:        
        test_history = trainer.evaluate(test_loader)
        print_evaluation(test_history)
    finally:
        del trainer
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()
