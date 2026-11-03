== Image Properties

This section looks at how properties such as brightness, contrast, sharpness etc. are split across the images and if there are any distinctive features about either fire or smoke to help distinguish from their surroundings. Two aspects are investigated, how these properties differ between fire, smoke and neither images, and how fire and smoke regions differentiate from their surrounding backgrounds.

This analysis also aims to identify any biases that could influence the model's performance. For example, if fire or smoke consistently appear under lighting conditions where the model may learn to associate these conditions with the presence of the object rather than relying on its visual features. Such relations could increase the chances of false positive and/or negatives when the model encounter different lighting conditions.

=== Property Distributions
#figure(
  image("../../../../results/plots/data_analysis/image_prop_box_plots.svg"),
  caption: [Property distribution across image types]
) <fig:image_props>

All images are converted to gray scale and properties are calculated via the pixel values.

// TODO reference the sections
==== Brightness <sect:image_brigthness>
Brightness is calculated as the mean pixel value across an image. Interestingly, between images that have no annotations and those that contain smoke there is a similar distribution while fire, which is something one would consider as bright, is seen as dark. This can be due to a multitude of reasons; the pictures are naturally darker, the exposure of the camera reducing the brightness of the fire to capture the rest of the image and, seen in @sect:bound_boxes, the fire boxes are generally tiny, so their brightness values across an entire image may not contribute as much to the overall mean, because it only takes up a small portion of the image. Further analysis will be done later to analyse any potential hypothesis for the reasons mentioned above.

==== Contrast
Contrast is calculated as the standard deviation of the pixel values across the image. The same trends can be seen as in @sect:image_brigthness. The reason for examining contrast is to understand how pixel values deviate from the mean image brightness and investigate whether the size of fire bounding boxes correlates with overall image contrast, particularly in cases where small fire regions occupy only a fraction of the image or larger fire regions contrast with darker backgrounds.

==== Sharpness
Sharpness is calculated as the variance of the Laplacian, which is the second order spatial differential operator of pixel intensities: $"Sharpness" = 1/N sum^N_(i=1)(L_i - mu)^2$, where $L_i = nabla^2f(x, y)$ and $mu = "avg"(L)$. This looks at how much pixel intensities change spatially, particularly around edges and fine details, but can also be influenced by the noise and texture of the image. Applying it to this case, it would generally be expected that smoke would have quite a low sharpness due to being highly diffuse, and fire to be quite high in sharpness. However, it also depends on the diffuseness of the smoke. From @fig:image_props, smoke tends to have higher sharpness, but this can be due to reasons such as vegetation and other objects being visible through the smoke, meaning the image is not completely obscured. Also, contributions from other parts of the image can influence this. For similar reasons to those mentioned in @sect:image_brigthness, the edges of the fire may cause higher sharpness, but if most of the image is quite uniform, such as a wall, and the fire is small, the overall sharpness of the image would not be that high. 

=== Properties Across Splits
#grid(
  columns: (40%, 60%),
  column-gutter: 2em,

  [
  The properties are then examined at the split-level in @fig:image_props_split to ensure that no single split contains disproportionately bright images, images with high contrast, or images with high levels of detail as measured by sharpness, but rather that these properties are similarly distributed across the dataset splits. Significant differences between splits could introduce unintended biases, where the model may perform better on certain splits due to differences in image characteristics rather than its ability to detect fire or smoke. Therefore, maintaining similar distributions helps ensure that the validation and test sets provide a more representative evaluation of the model's performance.
  ],
  [
    #figure(
      image("../../../../results/plots/data_analysis/image_props_by_split.svg", width: 115%, height: 35%),
      caption: [Image properties by split]
    ) <fig:image_props_split>
  ]
)

=== Properties of Annotations <sect:annotation_property>
#grid(
  columns: (1fr, 1fr),

  [
    #figure(
      image("../../../../results/plots/data_analysis/fire_only_properties.svg"),
      caption: [Fire boxes versus their background]
    ) <fig:fire_box_props>
  ],

  [
    #figure(
      image("../../../../results/plots/data_analysis/smoke_only_properties.svg"),
      caption: [Smoke boxes versus their background]
    ) <fig:smoke_box_props>
  ]
)

As discussed previously in @sect:image_brigthness within those plots the properties were looked at at a global image level, so it did not provide much information as to how these properties related to these bounding boxes. Properties within @fig:fire_box_props and @fig:smoke_box_props are only looked at within a single class type images due to the effects that terms like sharpness can have on boxes especially where fire and smoke may influence one another due to the nature of the items. 

Within @fig:fire_box_props it is still seen that fire itself is generally quite dark where the image around the fire tends to be even darker. This raises the question of whether it is exposure, the time of day or the effect of tiny annotations seen in @sect:bound_boxes. This would require a lot more analysis into whether its actually night time, whether these small annotations are very far away or small fires which we believe to be beyond the scope of this project.