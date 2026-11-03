= Network Structure

Our network structure consists of a ResNet @he2016resnet backbone and a YOLOv8 @ultralytics_yolov8 neck and head. The ResNet backbone is built layer-by-layer. The YOLOv8 neck is wired together by us from the Ultralytics Conv and C2f blocks, while the head is imported as a module from the Ultralytics library. The model also borrows its loss from the Ultralytics library.

We chose this model as, given the dataset size, it was useful to be able to initialise the ResNet backbone with ImageNet pretrained weights. We chose to combine this with a YOLO neck and head as it is suited to the wide range of object sizes in our images (both small fires and large smoke regions).

#figure(
  image("../../../images/hybrid_architecture.png", width: 80%),
  caption: [Our hybrid model],
) <fig:hybrid>





