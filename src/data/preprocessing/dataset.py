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
            flip_prob = 0.0,
            zoom_prob = 0.0,
            zoom_max = 1.3
        ):
        self.root = root
        self.data = data
        self.transform = transform

        self.flip_prob = flip_prob
        self.zoom_prob = zoom_prob
        self.zoom_max = zoom_max

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

        # Random zoom-in
        if torch.rand(1).item() < self.zoom_prob:
            zoom = 1.0 + torch.rand(1).item() * (self.zoom_max - 1.0)

            _, H, W = img.shape

            crop_h = max(1, round(H / zoom))
            crop_w = max(1, round(W / zoom))

            # randomally chosen top left to start from
            top = torch.randint(0, H - crop_h + 1, (1,)).item()
            left = torch.randint(0, W - crop_w + 1, (1,)).item()

            img = F.resized_crop(
                img,
                top=top,
                left=left,
                height=crop_h,
                width=crop_w,
                size=[H, W],
                antialias=True,
            )

            if len(labels) > 0:
                # Convert normalised YOLO boxes into pixel xyxy coordinates
                xyxy = torch.stack([
                    (labels[:, 1] - labels[:, 3] / 2) * W,
                    (labels[:, 2] - labels[:, 4] / 2) * H,
                    (labels[:, 1] + labels[:, 3] / 2) * W,
                    (labels[:, 2] + labels[:, 4] / 2) * H,
                ], dim=1)

                # Translate coordinates into the cropped image
                xyxy[:, [0, 2]] -= left
                xyxy[:, [1, 3]] -= top

                # Clip boxes to crop boundaries
                xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, crop_w)
                xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, crop_h)

                # Remove boxes that are entirely outside the crop
                valid = (
                    (xyxy[:, 2] > xyxy[:, 0]) &
                    (xyxy[:, 3] > xyxy[:, 1])
                )

                labels = labels[valid]
                xyxy = xyxy[valid]

                # Convert back to normalised YOLO coordinates
                labels[:, 1] = (xyxy[:, 0] + xyxy[:, 2]) / (2 * crop_w)
                labels[:, 2] = (xyxy[:, 1] + xyxy[:, 3]) / (2 * crop_h)
                labels[:, 3] = (xyxy[:, 2] - xyxy[:, 0]) / crop_w
                labels[:, 4] = (xyxy[:, 3] - xyxy[:, 1]) / crop_h

        return img, labels