import torch
from torchvision.transforms import functional as F
from torch.utils.data import Dataset

from PIL import Image

class ModelData(Dataset):
    def __init__(
            self, 
            root, 
            data, # tuple[paths, coordinates] 
            transform,
            flip_prob: float = 0.0,
        ):
        self.root = root
        self.data = data
        self.transform = transform

        self.flip_prob = flip_prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        path, boxes = self.data[idx] 
        img = Image.open(self.root / path).convert("RGB")

        img = self.transform(img)

        # (N, 5): class, x_center, y_center, width, height (normalised 0-1)
        # reshape keeps images with no boxes as (0,5) instead of (0,)
        labels = torch.tensor(
            [[box.category, *box.coordinates] for box in boxes],
            dtype=torch.float32,
        ).reshape(-1, 5)

        # Random horizontal flip
        if torch.rand(1).item() < self.flip_prob:
            img = F.hflip(img)

            labels[:, 1] = 1.0 - labels[:, 1]

        return img, labels