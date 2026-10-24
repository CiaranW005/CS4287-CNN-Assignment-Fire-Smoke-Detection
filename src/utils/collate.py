import torch

def collate_fn(batch):
    """Custom collate function for non-tensor items (boxes)

    Returns:
        images: torch.Tensor
        boxes: list[BoundingBox]
    """
    images, boxes = zip(*batch)

    images = torch.stack(images)

    return images, boxes