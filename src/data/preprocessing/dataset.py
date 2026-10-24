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
        path, boxes = self.data[idx] # TODO ensure coords are in the correct format for going through
        img = Image.open(self.root / path).convert("RGB")

        # TODO also transform coordinates
        img = self.transform(img)

        return img, boxes