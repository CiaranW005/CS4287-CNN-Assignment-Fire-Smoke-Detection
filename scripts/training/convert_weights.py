from pathlib import Path
import torch
from torchvision.models import ResNet50_Weights

from src.fine_tune.models.resnet import ResNetBackbone

from src.config.paths import WEIGHTS_DIR

"""
python -m scripts.training.convert_weights
"""

def map_resnet_weights(state_dict):
    mapped = {}

    for key, value in state_dict.items():
        if key.startswith("fc."): # Skips weights in the final classifier
            continue

        if key.startswith("conv1."):
            key = key.replace("conv1.", "stem.conv.", 1)

        elif key.startswith("bn1."):
            key = key.replace("bn1.", "stem.batch.", 1)

        else:
            for i in range(1, 5):
                key = key.replace(f"layer{i}.", f"stage{i}.", 1)

            key = key.replace(".bn1.", ".batch1.")
            key = key.replace(".bn2.", ".batch2.")
            key = key.replace(".bn3.", ".batch3.")
            key = key.replace(".bn4.", ".batch4.")
            key = key.replace(".downsample.", ".shortcut.")

        mapped[key] = value

    return mapped

if __name__ == "__main__":
    weights = ResNet50_Weights.IMAGENET1K_V2

    state_dict = torch.hub.load_state_dict_from_url(
        weights.url,
        map_location="cpu"
    )

    mapped_weights = map_resnet_weights(state_dict)

    # Verify compatibility before saving
    backbone = ResNetBackbone()
    backbone.load_state_dict(mapped_weights, strict=True)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(mapped_weights, WEIGHTS_DIR / "resnet50.pt")