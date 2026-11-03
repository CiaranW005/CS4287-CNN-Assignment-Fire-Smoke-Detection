== Model Results

The model with hyperparameters mentioend in @sec:hyperparameters performed well throughout training, as shown in @fig:train_val_loss. However, there are indications of potential overfitting, as training loss continued to decrease while validation loss began to plateau, resulting in an increasing gap between the two. Despite this divergence, validation mAP metrics continued to improve, suggesting that the model was still improving its detection performance even though the validation loss showed little further reduction. This highlights that improvements in detection performance do not necessarily correspond directly to reductions in validation loss. This discrepancy could be attributed to differences between the YOLO loss function and mAP evaluation. While YOLO loss combines multiple optimisation objectives, including bounding-box regression and classification, mAP measures detection performance through confidence rankings and bounding-box overlap at different IoU thresholds. Consequently, improvements in detection performance may not correspond directly to reductions in the combined loss.

#figure(
  image("../../../../../results/plots/training_validation_loss.svg"),
  caption: [Training and validation loss across epochs.],
) <fig:train_val_loss>

On the test set, our model achieved an mAP50 of 0.7414 and an mAP50-95 of 0.4070, as shown in @tab:test_overall. The difference between these metrics indicates that while the model can detect and localise a reasonable proportion of objects at an IoU threshold of 0.50, performance decreases when stricter localisation requirements are introduced. This suggests that precise bounding-box localisation remains an area for improvement.

#figure(
  table(
    columns: 2,
    align: (left, right),
    table.header([*Metric*], [*Score*]),
    [mAP50], [0.7414],
    [mAP50-95], [0.4070],
  ),
  caption: [Overall detection performance on the test set.],
) <tab:test_overall>

#figure(
  table(
    columns: 5,
    align: (left, right, right, right, right),
    table.header([*Object size*], [*mAP50-95*], [*mAR*], [*Recall50*], [*Recall75*]),
    [Overall], [0.4070], [0.5492], [0.9042], [0.5614],
    [Small], [0.2708], [0.4104], [0.8272], [0.3429],
    [Medium], [0.4044], [0.5480], [0.9205], [0.5670],
    [Large], [0.4727], [0.6064], [0.9248], [0.6530],
  ),
  caption: [Test-set detection performance by object size. Recall50 and Recall75 use IoU thresholds of 0.50 and 0.75, respectively.],
) <tab:test_size>

As shown in @tab:test_size, detection performance improves with object size across all four metrics. Small objects achieved the lowest mAP50-95 of 0.2708, compared with 0.4044 for medium objects and 0.4727 for large objects. This reinforces the concerns identified during the experiments regarding the difficulty of detecting smaller objects.

A particularly significant difference can be observed in recall. For small objects, recall decreases from 0.8272 at IoU 0.50 to 0.3429 at IoU 0.75, a reduction of 48.43 percentage points. In comparison, large-object recall decreases from 0.9248 to 0.6530, a reduction of 27.18 percentage points. This suggests that while the model can identify many smaller objects at lower IoU thresholds, achieving sufficiently accurate bounding-box localisation becomes considerably more difficult as the IoU threshold increases.

#figure(
  table(
    columns: 2,
    align: (left, right),
    table.header([*Class*], [*mAP50-95*]),
    [Smoke], [0.4689],
    [Fire], [0.3451],
  ),
  caption: [Test-set average precision for each class, averaged over IoU thresholds from 0.50 to 0.95.],
) <tab:test_class>

As shown in @tab:test_class, smoke achieved a higher mAP50-95 of 0.4689 compared with 0.3451 for fire, representing a difference of 12.38 percentage points. This indicates that the model performs considerably better at detecting smoke than fire. Although the aggregate metrics do not establish the cause of this difference, the results identify fire detection as an area requiring further improvement.