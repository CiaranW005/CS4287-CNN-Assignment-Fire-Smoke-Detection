== Near-Duplicates in the Test Split

Using the same hash as in cross-validation, we see that 2,109 of the 4,306 test images (49%) have a near-duplicate in the train or validation split. These make the test score higher than it would be on new images, so @tab:test_subsets scores them separately.

#figure(
  table(
    columns: 6,
    [*Test images*], [*Images*], [*mAP50*], [*mAP50-95*], [*Smoke / Fire mAP50-95*], [*mAR100*],
    [All], [4,306], [0.741], [0.407], [0.469 / 0.345], [0.549],
    [No near-duplicate], [2,197], [0.699], [0.378], [0.463 / 0.292], [0.533],
    [With near-duplicate], [2,109], [0.859], [0.504], [0.491 / 0.518], [0.604],
  ),
  caption: [Test results with and without near-duplicates of train or validation images.],
) <tab:test_subsets>

Without near-duplicates, mAP50-95 drops from 0.407 to 0.378. The drop is mostly in fire, which falls from 0.345 to 0.292, while smoke only falls from 0.469 to 0.463. The validation split has the same problem, so the validation and test scores end up having similar results.
