== Transfer Learning <sec:transfer_learning>

Our ResNet-50 backbone is initialised with torchvision's 'IMAGENET1K_V2' weights @torchvision_resnet, which were trained to classify the 1.28 million images of ImageNet into 1,000 classes. The backbone therefore already encodes general visual features such as edges, textures and shapes. Because our model only uses the backbone as a feature extractor, the final fully-connected classification layer is left out and only the stem and the four residual stages are used.

#figure(
  table(
    columns: 3,
    [*Component*], [*Initialisation*], [*Learning rate*],
    [ResNet-50 stem + stages 1–4], [ImageNet-1K V2], [$0.1 times eta$],
    [1×1 adapter convolutions], [Random], [$eta$],
    [YOLOv8 neck (FPN + PAN)], [Random], [$eta$],
    [YOLOv8 detection head], [Random], [$eta$],
  ),
  caption: [Initial learning rate of each component.],
) <tab:transfer_learning>

No layers of our model are frozen. We opted to instead train the model on a differential learning rate that applies a learning rate 10 times smaller than the rest of the model to only the backbone. This prevents the model from losing the feature knowledge it had already developed from training on ImageNet.

The $1 times 1$ adapter convolutions, the YOLOv8 neck and head all borrow only the YOLOv8 architecture and have randomly initialised weights, therefore they are trained from scratch at a full learning rate.

=== Weight Initialisation
The randomly initialised layers use PyTorch's default Kaiming uniform initialisation, which keeps activations from shrinking or growing through the network. The classification biases in the head are set by Ultralytics @ultralytics_yolov8 so that every class starts with a confidence close to 0. As most of the image is background, this stops the model from making large errors in the first epochs.