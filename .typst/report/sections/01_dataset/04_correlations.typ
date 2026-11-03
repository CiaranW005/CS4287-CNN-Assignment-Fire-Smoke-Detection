== Correlations

This last section examines how different characteristics of the dataset relate to one another to check whether there are any linear or monotonic relationships within the dataset. For this analysis, we use Pearson's correlation coefficient to examine linear relationships, while Spearman's rank correlation coefficient is used to identify monotonic relationships. For example, this can include correlations between the number of bounding boxes and the size of an image, or how bounding-box coverage relates to global image properties such as brightness, contrast, and sharpness. Additionally, correlations between the properties of bounding boxes and their surrounding backgrounds can be examined, such as whether increases in bounding-box brightness are associated with changes in background brightness.

=== Box area compared to Box Count and Image Area
#figure(
  image("../../../../results/plots/data_analysis/correlations.svg", height: 30%),
  caption: [Correlation coefficients of box counts/image area vs box area]
) <fig:box_image_area_corrleations>

Box area is measured as the relative area of an image a box occupies in an image. The usefulness of @fig:box_image_area_corrleations is that it let us decide in preprocessing the downsizing of images. With the figure it can be seen that each box area tends to decrease when the image area gets larger which can be a problem when small boxes get compressed, leading to the loss of information. For example, they could be $20 times 20$ pixels in their original image and could go as small as $2 times 2$ pixels within the new image size for preprocessing. 

Within box counts another relationship that can be seen is between the min and max areas. As the number of boxes increase, the minimum bounding box area tends to decrease, while the maximum area tends to increase. This could be due to differences in annotation practice, where some objects may be annotated using tiny bounding boxes or due to the nature of the images containing objects of various sizes. As discussed in @sect:box_cleaning, zero-area bounding boxes were removed during data cleaning. However, some miniscule annotations may still represent annotation errors or objects that contain very little useful information after preprocessing. Determining whether these annotations are valid would require further manual inspection which is beyond the bandwidth of this project.

=== Annotations compared to their backgrounds
#align(center)[
  #box(width: 120%)[
    #grid(
  columns: (1fr, 1fr),
  column-gutter: 10pt,

  [
    #figure(
      image("../../../../results/plots/data_analysis/fire_only_correlations.svg", width: 100%),
      caption: [
        Correlation coefficients for fire-only boxes \
        (a) Bounding box properties vs global Properties \
        (b) Bounding box coverages vs global properties deltas
      ] 
    ) <fig:fire_corr>
  ],
  [
    #figure(
      image("../../../../results/plots/data_analysis/smoke_only_correlations.svg"),
      caption: [
        Correlation coefficients for smoke-only boxes \
        (a) Bounding box properties vs global Properties \
        (b) Bounding box coverages vs global properties deltas
      ]
    ) <fig:smoke_corr>
  ]
    )
  ]
]

Within @sect:annotation_property it was seen there was a gap between the bounding box annotation properties and the background of those images. @fig:fire_corr and @fig:smoke_corr aim to help quantify these differences in seeing how strongly the properties of fire and smoke regions correlate with those of their surrounding backgrounds. They also examine the coverage of these bounding boxes and how their properties change relative to their background.

In @fig:fire_corr, there is a strong positive correlation between the brightness of fire bounding boxes and their surrounding backgrounds, with Pearson (0.75) and Spearman (0.76), respectively, indicating that both tend to increase and decrease together. As hypothesised in @sect:annotation_property, one possible explanation is that many images were taken at night. However, this correlation alone cannot determine whether that is the case, as it could also be caused by fire illuminating its surroundings or bounding boxes not perfectly fitting the fire and including some background pixels.\
Another possible explanation mentioned was camera exposure, which could influence the properties of the image depending on how the photograph was taken. A strong Pearson correlation of 0.81 is observed between fire and background sharpness, although this weakens to a moderate Spearman correlation of 0.49, potentially indicating that extreme values influence the Pearson coefficient. It was also hypothesised that brighter fires could cause overexposure or blur, reducing the sharpness of their surroundings. However, this is not strongly supported by the correlation between fire brightness and background sharpness, with Pearson (0.03) and Spearman (-0.14), indicating little to no consistent relationship across the dataset.\
Coverage-wise, fire bounding boxes generally do not show strong relationships with differences in their background properties, with the most apparent relationship being sharpness, with Pearson (0.28) and Spearman (0.44), indicating a weak to moderate positive relationship. However, when examining absolute differences, these correlations weaken to the point where there is little to no apparent relationship. The difference measures how much a property increases or decreases between the bounding box and its background, while the absolute difference measures the magnitude of this change regardless of its direction. This suggests that greater fire coverage may be associated with the direction of the sharpness difference, rather than necessarily producing larger differences in sharpness.

In @fig:smoke_corr, most relationships are weak to moderate, except for sharpness, which has a Pearson correlation of 0.75 and a Spearman correlation of 0.68, indicating a moderate to strong positive relationship between smoke-region and background sharpness. However, this does not reveal much about the underlying cause, as the relationship could exist for several reasons, such as smoke being partially transparent and allowing background details to remain visible, or both regions sharing similar image characteristics. Therefore, while a strong relationship is observed, further analysis would be required to determine what contributes to it.

Coverage-wise, the only relationship of interest is contrast, with Pearson (0.43) and Spearman (0.52), indicating a moderate positive correlation between smoke coverage and the difference in contrast between smoke and its background. This suggests that as smoke occupies a larger proportion of an image, its contrast relative to the background tends to increase, which may be related to the varying density and transparency of smoke. As discussed previously regarding sharpness, these properties may also be influenced by the environment and conditions in which an image was captured, rather than smoke alone.

