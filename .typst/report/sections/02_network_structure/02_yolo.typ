== Adapter 

ResNet's channel counts are larger than the YOLO neck expects so to connect the backbone to the neck, we implemented an adapter. It consists of a $1 times 1$ convolution, followed by batch normalisation and the SiLU activation function, which brings the channels down to $"widths" = (128, 256, 512)$ to match what the YOLO neck expects.

== YOLOv8 Neck
In CNNs, the model shrinks the image as it goes deeper into the model which can result in early, large maps that know _where_ objects are in the image but not _what_ they are. Deeper layers with smaller maps will know what things are but only roughly where.
For the model to learn better as a whole, we mimicked the architecture of modern YOLO models and used the FPN @lin2017fpn top-down and PAN @liu2018panet bottom-up structure for the neck of the model.

The neck first completes a top-down pass, which upsamples smaller, deeper layers to match the size of the next, higher-resolution map (@fig:neck_topdown). It then concatenates the two maps. Upsampling happens twice before the model moves to the PAN pass.

#figure(
  image("../../../images/YOLOneck_topdown.png", width: 80%),
  caption: [YOLOv8 neck: the top-down FPN pathway]
) <fig:neck_topdown>

A bottom-up pass is then completed, where the earlier, finer maps are downsampled to the size of the next, lower-resolution map and concatenated with the smaller map (@fig:neck_bottomup). This also happens twice before the model moves to the head.

#figure(
  image("../../../images/YOLOneck_bottom-up.png", width: 80%),
  caption: [YOLOv8 neck: the bottom-up PAN pathway]
) <fig:neck_bottomup>


== YOLOv8 Head

The head of our model is a single Detect module imported from the Ultralytics library @ultralytics_yolov8, which has a separate set of layers for each of the three feature maps coming from the neck of the model. There are two separate branches per feature map. The first is a classification branch, which gives a confidence score for each class and the other is a box branch which predicts how far the object's left, top, right and bottom edges are from the centre of each grid cell.

#figure(
  image("../../../images/YOLO_head.png", width: 80%),
  caption: [Detect head: separate box and classification branches per feature map.]
) <fig:detect_head>