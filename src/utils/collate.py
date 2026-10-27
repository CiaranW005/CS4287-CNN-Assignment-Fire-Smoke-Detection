import torch

def collate_fn(batch):
    """Collates images and boxes into the format v8DetectionLoss expects

    Returns:
        images: torch.Tensor (B, 3, 640, 640)
        targets: dict
            batch_idx: (N, ) index of the image each box belongs to
            cls: (N, ) class of each box
            bboxes: (N, 4) x_center, y_center, width, height (normalised 0-1)
    """
    images, labels = zip(*batch)

    images = torch.stack(images)

    #prefix each box with the index of its image in the batch
    labels = torch.cat([
        torch.cat([torch.full((len(label), 1), i, dtype=label.dtype), label], dim=1)
        for i, label in enumerate(labels)
    ])

    return images, {
        "batch_idx": labels[:, 0],
        "cls": labels[:, 1],
        "bboxes": labels[:, 2:],
    }