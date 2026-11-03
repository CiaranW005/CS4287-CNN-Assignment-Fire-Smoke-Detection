== Bounding Boxes

This section examines the distribution and validity of bounding-box annotations throughout the dataset, including their spatial locations and the proportion of each image occupied by an average bounding box. An important part of this analysis is identifying invalid annotations, such as bounding boxes with zero area or coordinates extending beyond the image boundaries. These errors must be addressed to ensure that the annotations provide meaningful information during model training.

=== Annotation Validation and Cleaning <sect:box_cleaning>
The analysis identified 18 bounding boxes with zero area. Examples of these annotations are shown in @fig:error_examples. Since these boxes contain no spatial information about the annotated objects and appear to result from annotation errors, they were removed from the dataset to prevent potential issues during model training.

Additionally, 379 bounding boxes across 322 images were found to extend beyond the image boundaries. These annotations were corrected by clipping their coordinates to the valid image boundaries and recalculating their centres and dimensions. This ensures that the bounding boxes represent only regions within the image, preventing the model from learning invalid spatial coordinates. 

#figure(
  grid(
    columns: 2,

    image(
      "../../../../results/images/data_analysis/error_box_10499.png"
    ),

    image(
      "../../../../results/images/data_analysis/error_box_20960.png"
    ),
  ),
  caption: [Examples of images with invalid bounding-boxes (area=0)]
) <fig:error_examples>

=== Distribution of Bounding Boxes <sect:bound_boxes>

The distribution of bounding-box areas is heavily right-skewed, as shown in @fig:box_areas, with a high probability density for bounding boxes occupying a small proportion of their respective images. This density decreases rapidly as bounding-box area increases, although a long tail extends towards annotations occupying nearly the entire image. The probability density represents the concentration of bounding-box areas, where the area under the curve over a given interval represents the probability of an annotation falling within that range.

This is an important consideration during preprocessing, as reducing image resolution could result in already small annotated regions containing very little visual information, potentially making them more difficult for the model to detect.

Further analysis by category in @fig:fire_smoke_areas reveals substantial differences between fire and smoke annotations. Fire bounding boxes are heavily concentrated towards the lower end of the distribution, with a sharp peak near zero and a rapid decrease in density as the bounding-box area increases. In contrast, smoke bounding boxes have a much broader distribution, with a lower peak and a gradual decrease in density across a wider range of areas. Although both categories contain some exceptionally large bounding boxes, the majority of fire annotations occupy substantially smaller proportions of their images compared with smoke annotations.

These findings suggest that bounding-box size may influence detection performance, particularly for smaller fire annotations. During model evaluation, detection performance should therefore be examined across different bounding-box sizes to determine whether smaller objects are disproportionately difficult to detect and whether image preprocessing contributes to this limitation.

#figure(
  image("../../../../results/plots/data_analysis/box_area_distribution.svg", width: 85%),
  caption: [Kernel density estimation of relative bounding-box areas across the dataset.]
) <fig:box_areas>

#figure(
  image("../../../../results/plots/data_analysis/fire_vs_smoke_area_distribution.svg", width: 85%),
  caption: [Kernel density estimation of relative bounding-box areas for fire and smoke annotations.]
) <fig:fire_smoke_areas>

Finally, the distribution of bounding-box areas across the training, validation, and test splits was examined to determine whether each subset contains a representative range of bounding-box sizes. As shown in @fig:area_box_split, the distributions are broadly consistent across all three splits. Outliers were excluded from the visualisation to improve readability, allowing the general trends to be compared more clearly despite differences in the number of images in each split.

These findings suggest that no additional stratification or adjustments are necessary to ensure that bounding-box sizes are proportionally represented across the dataset splits.

#figure(
  image(
    "../../../../results/plots/data_analysis/area_dist_by_split.svg"
  ),
  caption: [Box Plots of area distribution by split]
) <fig:area_box_split>
=== Spatial Distribution of Bounding Boxes <sect:box_positions>
#grid(
  columns: (1fr, auto),
  column-gutter: 2em,

  [
    The spatial locations of the bounding boxes were examined by plotting the distribution of their centre coordinates, as shown in @fig:box_positions. Both fire and smoke annotations are concentrated towards the centre of their images, with smoke peaking slightly above the centre and fire spreading more evenly around it. Horizontally, both distributions are roughly symmetric, indicating no strong bias towards either side of an image. Small clusters of fire boxes can also be seen at the far left and right edges, which are likely objects cut off by the image boundaries.

    This central concentration suggests that the model could learn to associate fire and smoke with the centre of an image rather than relying on their visual features, which is considered when selecting the augmentations in @sect:rand_augs.
  ],
  [
    #figure(
      image("../../../../results/plots/data_analysis/box_centre_positions.png", height: 10cm),
      caption: [Distribution of bounding-box \ centres for smoke and fire]
    ) <fig:box_positions>
  ]
)
