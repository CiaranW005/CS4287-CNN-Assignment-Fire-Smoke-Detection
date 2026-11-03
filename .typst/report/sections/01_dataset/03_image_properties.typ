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
Brightness calculated as the mean pixel value across an image. Interestingly between images that have no annotations and smoke there is a similar distribution across these images but strangely fire which is someone would consider as bright especially compared to smoke is seen as dark this can be due to a multitude of reasons the pictures are naturally darker, the exposure of the camera reducing the brightness of the fire to capture the rest of the images and seen in @sect:bound_boxes that the fire boxes are generally tiny, so their brightness values across an entire image may not contribute as much to the overall mean, because it only takes up small portion of the image. Further analysis will be done later to analyse any potential hypothesis for the reasons mentioned above.

==== Contrast
Contrast calculated as the standard deviation of the pixel values across the image. The same trends can be seen as in @sect:image_brigthness. The reason for examining contrast is to understand how pixel values deviate from the mean image brightness and investigate whether the size of fire bounding boxes correlates with overall image contrast, particularly in cases where small fire regions occupy only a fraction of the image or larger fire regions contrast with darker backgrounds.

==== Sharpness
Sharpness calculated as the variance of the Laplacian which is the second order spatial differential operator of pixel intensities. $"Sharpness" = 1/N sum^N_(i=1)(L_i - mu)^2 -> L_i = nabla^2f(x, y) "and where" mu = "avg(L)"$. This looks at how much pixel intensities changes spatially and more so around edges and fine details, but can also be influenced by noise and texture of the image. So applying it to this case it would be generally expected smoke to have quite a low sharpness due to being highly diffuse and fire to be quite high in sharpness. However, it also depends on the diffuseness of the smoke from @fig:image_props smoke tends to have larger sharpness, but this can be due to reasons such as vegetations and other objects being visible through the smoke making it not entirely impossible to see what is in the images also contributions from outside the image can influence this also, and due to similar reasons mentioned in @sect:image_brigthness where the edge of fire to cause higher sharpness but if most of the image is quite uniform in feature such as a wall and likewise and the picture of the fire is small the sharpness of the images would not be that high. 

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

As discussed previously in @sect:image_brigthness within those plots the properties were looked at a global image level, so it did not provide much information in how these properties related to these bounding boxes. Properties within @fig:fire_box_props and @fig:smoke_box_props are only looked at within a single class type images due to the effects that terms like sharpness can have on boxes especially where fire and smoke may influence one another due to the nature of the items. 

Within @fig:fire_box_props it is still seen fire itself is generally quite dark where the image around the fire tends to be even darker. This raises the question to whether is it exposure, is the time of day or is it the effect of tiny annotations seen in @sect:bound_boxes. This would require a lot more analysis into whether its actually night time, whether these small annotations are very far away or small fires which we believe to be beyond the scope of this project.