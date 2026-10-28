from torchvision import transforms

# resize only so the normalised box coordinates stay valid
train_transform = transforms.Compose([
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.015, # small hue shift so fire keeps its color (YOLO default)
    ),
    transforms.ToTensor(),
    # bacbone is pretrained in ImageNet so keep its normalisation
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # Colour channels from ImageNet
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])