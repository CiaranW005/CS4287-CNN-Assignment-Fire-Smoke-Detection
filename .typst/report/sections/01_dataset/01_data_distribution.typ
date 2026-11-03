==  Sample Distribution

This section examines the overall composition of the dataset in terms of its train/val/test split and the proportion of samples within each split and the number of bounding boxes for each category, fire (1) and smoke (0). The distributions help see whether there are any class biases within the dataset and if any stratification or changes are needed to balance out the subsets. This is important for identifying potential class imbalances that could bias the model towards more frequently represented classes, reducing its ability to detect underrepresented categories.

#figure(
  image("../../../../results/plots/data_analysis/data_split.svg", width: 55%),
  caption: [Distribution of images across the training, validation, and test splits.]
) <fig:data_split>

As seen in @fig:data_split the dataset from Kaggle is already pre-split into predefined training, validation and test subsets. However, for future cross-fold validation, train and validation subsets combined contain \~17,000 images with validation making up \~18% of that, so a five-fold cross-validation can help keep that split in the data.

Next is examining how the images are annotated within these splits @fig:class_distribution. There is overall an even-distribution of images containing fire and/or smoke (54.3%) versus containing neither (45.7%). This suggests that no additional balancing is needed between these two categories.

However, the second pie-chart which observes only the images containing fire and/or smoke, initially suggests a significant class imbalance, with smoke-only images accounting for 50.2% of the images compared to just 10% containing just fire. When considering both class, roughly 90% of the images contain at least one smoke-annotation. Further analysis of the bounding-box annotation is needed to determine whether this imbalance is also seen in the number of annotations of each class. 

#figure(
  image("../../../../results/plots/data_analysis/image_category_distribution.svg", width: 95%),
  caption: [
    Distribution of fire and smoke images: \
    (a) Proportion of images containing fire or smoke versus neither. \
    (b) Distribution of smoke-only, fire-only and combined images
  ]
) <fig:class_distribution>

The next stage now looks at how these annotations aredistributed across the training, validation and test splits. This is to ensure that each split gets an even sample of images such that there are no biases in that data at this level. For example, if the training set contained mostly smoke-only images while the test set has mostly fire-only, the model could struggle to detect fire due to insufficient exposure to representative examples during training.

However, in the actual data shown in @fig:image_distirbution_split this is not a problem as each split maintains across all three splits, which is distributed to; Both(\~20%), Smoke-only(\~30%), Fire-only(\~5%), None(\~45%). This suggests that the predefined splits are well-balanced relative to the overall dataset composition, but will need to be kept in mind for cross-fold validation to maintain these compositions.

#figure(
  image("../../../../results/plots/data_analysis/category_distribution_by_split.svg", width: 55%),
  caption: [
    Normalised proportions of image-annotations across splits
  ]
) <fig:image_distirbution_split>

#grid(
  columns: (1fr, 1fr),

  [
    As mentioned in the discussion around @fig:class_distribution there could be a problem with the distribution of fire and smoke annotations. However, @fig:fire_vs_smoke reveals the opposite of this trend is true where there tends to be more fire annotations (55.3%) compared to smoke (44.7%). This indicates that although smoke appears in more images, fire is annotated more frequently overall. 

    This is not a huge difference and there is no reason to try and enforce this to an exact 50-50 split. Such changes could cause a loss of valuable samples and reduce the variety of contexts represented within the dataset.
  ],

  [
    #figure(
    image("../../../../results/plots/data_analysis/label_distribution.svg", width: 80%),
    caption: [Fire vs Smoke box annotations]
  ) <fig:fire_vs_smoke>
  ]
)

