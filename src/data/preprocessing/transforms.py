from torchvision import transforms

# TODO find the proper way to transform for putting into Resnet/YOLO
# Transform from bursary will probably need to be changed
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
    ),
    transforms.ToTensor(),
    # Change this based on weights chosen
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # Colour channels from ImageNet
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])