= Preprocessing

From both the @sect:eda and experimentation we decided on this preprocessing pipeline for our data. The data was cleaned for any mistakes in labelling discussed in @sect:box_cleaning

== Resizing and Letterboxing
Letterboxing is a technique recommended by Ultralytics@ultralytics_preprocessing as it helps maintain the aspect ratios and proportions of objects when images are resized. This introduces padding around images when their aspect ratios differ from the designated input dimensions. One potential problem is that smaller objects may lose further visual information during down scaling, particularly when their dimensions are already small. However, experimenting with and without letterboxing showed improvements in small-object detection, despite some reductions in overall model performance. Therefore, we decided to use letterboxing and resize all images to a fixed resolution of $768 times 768$.

== ColourJitter <sec:colour_jitter>
We randomly augment the brightness, contrast, saturation and hue of images using torchvision's ColourJitter within the training set. This introduces variations in image properties during training, allowing the model to generalise better across different lighting conditions and environments. The selected values determine the range from which the augmentation factors are randomly sampled.

- *Brightness = 0.2 [0.8, 1.2]:* Randomly adjusts the brightness of an image. This is particularly useful for fire images, which were generally found to be darker during EDA. Randomly brightening or darkening these images can help reduce the model's reliance on specific lighting conditions.

- *Contrast = 0.2 [0.8, 1.2]:* Randomly adjusts the differences between brighter and darker regions of an image. This can help the model generalise across environments where fire and smoke may appear more or less distinguishable from their backgrounds.

- *Saturation = 0.2 [0.8, 1.2]:* Randomly adjusts the intensity of colours within an image. This helps account for differences in camera characteristics and environmental conditions, particularly where fire and smoke may appear with different colour intensities.

- *Hue = 0.015 [-0.015, 0.015]:* Randomly shifts the hue within torchvision's normalised hue range. This value is the default hue augmentation parameter used by Ultralytics YOLO @ultralytics_augmentation. The relatively small adjustment helps preserve the general colours of fire and smoke while introducing some colour variation.

== Normalisation

The images are then normalised using the standard RGB channel statistics from ImageNet, which is the dataset used to pretrain our ResNet backbone. The mean values for these channels are [0.485, 0.456, 0.406], with standard deviations of [0.229, 0.224, 0.225] @torchvision_resnet, respectively. Normalising using these values helps maintain consistency with the input distribution used during pretraining, allowing the model to make better use of its pretrained weights rather than having to adapt to an entirely different input scale.

== Random Augmentations <sect:rand_augs>

Two random augmentations introduced into our pipeline to help improve the generalisation of our model were horizontal flips and random zoom-ins. Boundary boxes were also updated relative to whenever each augmentation was applied.

// TODO @emmao1024 do you have that bounding box image thing where they were localised to add about this
- *Horizontal Flips (p = 50%)* - This augmentation randomly flips images horizontally, helping prevent the model from becoming overly reliant on the horizontal positioning of objects and backgrounds. As discussed in [EDA], this encourages the model to recognise fire and smoke under different spatial arrangements rather than associating them with particular sides of an image.

- *Random Zoom-ins (p = 50%)* - As discussed in @sect:bound_boxes, fire bounding boxes generally occupied a very small proportion of their images. To help address this, random zoom-ins were introduced to increase the apparent size of smaller annotations during training, potentially making them easier for the model to learn from. The maximum zoom-in factor was 30%, bounding boxes were clipped or removed if they were no longer in this new image.