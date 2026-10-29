= Network Structure

Our network structure consists of a ResNet backbone and a YOLOv8 neck and head. Both the ResNet backbone and YOLOv8 neck are built layer-by-layer, while the head is imported as a module from the Ultralytics library. The model also borrows its loss from the Ultralytics library.






