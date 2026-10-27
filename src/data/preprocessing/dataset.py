import torch
from torch.utils.data import Dataset
from PIL import Image

class ModelData(Dataset):
    def __init__(
            self, 
            root, 
            data, # tuple[paths, coordinates] 
            transform
        ):
        self.root = root
        self.data = data
        self.transform = transform

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

        return img, labels