== Cross-Validation

Some of the images in the dataset are near-identical frames of the same scene. If one of these is in the training fold and another in the validation fold, the model is tested on an image it has already seen which can cause data leakage. To avoid this, each image is given a difference hash (dHash), and images whose hashes differ by 4 bits or less are placed in the same group. The groups are then split across five folds, so a group is never in two folds. The original splits do not do this, as 1,459 of the 3,099 validation images share a group with a training image.

#figure(
  table(
    columns: 6,
    [*Fold*], [*Images*], [*Groups*], [*Empty images*], [*Smoke boxes*], [*Fire boxes*],
    [0], [3,445], [1,922], [47.5%], [1,901], [2,095],
    [1], [3,444], [1,922], [45.0%], [2,002], [2,134],
    [2], [3,444], [1,925], [38.2%], [2,074], [3,017],
    [3], [3,444], [1,923], [55.6%], [1,611], [2,212],
    [4], [3,444], [1,926], [41.1%], [1,955], [2,349],
  ),
  caption: [Composition of each cross-validation fold.],
) <tab:fold_balance>

The folds have the same size but differ in content (@tab:fold_balance). Fold 2 has the most fire boxes and the fewest empty images while fold 3 has the most empty images.

To keep training time down, each run trains for 8 epochs on 3,000 images from four folds and is validated on 1,000 images from the fifth. The other settings match the full training (batch size of 8 with 4 accumulation steps, 768 pixel images, AdamW optimiser). One epoch took about 97 seconds on an RTX 3080 (10GiB VRAM variant).

We cross-validated the base setting and a weight decay of $10^(-3)$, the only change that beat the base in hyperparameter tuning (@tab:tuning).

#figure(
  table(
    columns: 4,
    [*Fold*], [*Base*], [*Weight decay $10^(-3)$*], [*Difference*],
    [0], [0.112], [0.124], [+0.012],
    [1], [0.112], [0.113], [+0.001],
    [2], [*0.151*], [0.141], [−0.011],
    [3], [0.125], [0.121], [−0.005],
    [4], [0.133], [0.137], [+0.005],
    [*Mean ± std*], [*0.127 ± 0.017*], [*0.127 ± 0.012*], [+0.0005],
    [mAP50 mean], [0.336], [0.336], [0.000],
    [recall50 mean], [0.861], [0.856], [−0.005],
    [mAR100 mean], [0.431], [0.428], [−0.003],
  ),
  caption: [Best validation mAP50-95 on each fold, with means of other metrics.],
) <tab:cross_validation>

Both settings perform the same (@tab:cross_validation). Their mean mAP50-95 differs by only 0.0005 and a paired t-test gives $p = 0.90$. Weight decay $10^(-3)$ performs its best at fold 0, while the base does best on fold 2.

Both settings score highest on fold 2, which has the most fire boxes and the fewest empty images, and lowest on fold 1. In 9 of the 10 runs the best epoch was the last one so the model was still improving when training stopped (@fig:cross_validation). Training for longer likely has more influence than the choice of weight decay.

#figure(
  image("../../../../../results/plots/cross_validation/base_folds.svg", width: 100%),
  caption: [Training (solid) and validation (dashed) loss, and validation mAP50-95, for each fold of the base setting.],
) <fig:cross_validation>
