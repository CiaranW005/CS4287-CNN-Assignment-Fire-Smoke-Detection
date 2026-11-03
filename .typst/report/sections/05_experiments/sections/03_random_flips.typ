== Random Horizontal Flips <sect:rand_flips>

// TODO reference heat map of bounding box placements
Random horizontal flips were added to the augmentation pipeline to investigate whether changing the orientation of an image could improve model performance. As mentioned in @sect:rand_augs, horizontal flipping has a 50% probability of being applied to each image during training, potentially allowing the model to generalise better to different spatial arrangements of fire and smoke annotations.

The parameters of the run were as follows:

#figure(
  table(
    columns: 3,
    [*Image Size*], [*Batch Size*], [*Accum Steps*],
    [$640 times 640$], [14], [2]
  ),
  caption: [Training parameters for the random horizontal flip experiment.],
)

The experiment achieved the following results, including the best small-object mAP50-95, which evaluates objects with an area smaller than $32^2$ pixels, as mentioned in @sec:model_compare:

#figure(
  table(
    columns: 5,
    [*Epochs*], [*Best Epoch*], [*Best mAP50*], [*Best mAP50-95*], [*Best Small mAP50-95*],
    [18], [16], [0.721], [0.391], [0.227]
  ),
  caption: [Detection performance with random horizontal flipping.],
)

Adding random horizontal flipping improved the results compared with the previous experiment in @sect:lr_plateau, which achieved a best mAP50-95 of 0.363. This represents an improvement of 2.8 percentage points, in mAP50-95. The improvement suggests that introducing spatial variation through horizontal flipping can help the model generalise better to different object orientations and positions. Due to the improvement observed, this augmentation was selected for inclusion in the final training configuration. Despite the improvement in overall performance, small-object mAP50-95 remained considerably lower at 0.227 compared with the overall mAP50-95 of 0.391. This reinforces the concerns identified during EDA regarding the large number of small fire annotations and suggests that further experimentation should investigate how preprocessing and augmentation affect small-object detection.