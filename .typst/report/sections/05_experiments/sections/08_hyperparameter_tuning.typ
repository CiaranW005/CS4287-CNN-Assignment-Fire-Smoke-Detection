== Hyperparameter Tuning <sect:hyperparams>

For the base setting, we use a learning rate of $eta = 10^(-3)$, a weight decay of $10^(-4)$, a backbone learning rate of $0.1 times eta$ (@tab:transfer_learning) and augmentation turned on. Each experiment changes one of these per increment. All runs use the same training budget as cross-validation where validation occurs on fold 0 and the runs use the same seed.

#figure(
  table(
    columns: 6,
    [*Experiment*], [*Best epoch*], [*mAP50*], [*mAP50-95*], [*Smoke / Fire mAP50-95*], [*vs. base*],
    [Weight decay $10^(-3)$], [8], [0.356], [*0.131*], [0.138 / 0.124], [+0.012],
    [Base], [7], [0.334], [0.119], [0.129 / 0.109], [0],
    [Weight decay $10^(-5)$], [8], [0.335], [0.115], [0.130 / 0.100], [−0.004],
    [$eta = 3 times 10^(-3)$], [7], [0.300], [0.110], [0.126 / 0.094], [−0.009],
    [Augmentation off], [8], [0.305], [0.109], [0.093 / 0.124], [−0.011],
    [$eta = 3 times 10^(-4)$], [8], [0.283], [0.103], [0.117 / 0.089], [−0.016],
    [$eta = 2 times 10^(-3)$], [8], [0.297], [0.101], [0.095 / 0.108], [−0.018],
    [Backbone at $1 times eta$], [8], [0.265], [0.093], [0.080 / 0.106], [−0.026],
    [$eta = 10^(-4)$], [8], [0.216], [0.071], [0.084 / 0.059], [−0.048],
  ),
  caption: [Results of each tuning experiment on validation fold 0.],
) <tab:tuning>

The learning rate has the biggest effect on mAP50-95 (@tab:tuning). Higher and lower rates both do worse, and $10^(-4)$ is the worst run, as it may learn too slowly for 8 epochs.

Training the backbone at the full learning rate costs 0.026 mAP50-95, which reinforces using a lower learning rate for the pretrained backbone.

Turning augmentation off causes the model to overfit with a final gap (validation loss − training loss) of 14.1, against 4.8 for the base. Smoke drops from 0.129 to 0.093, while fire slightly improves.

A weight decay of $10^(-3)$ beat the base by 0.012. This is from one run on one fold, and cross-validation showed the two perform the same (@tab:cross_validation).

#figure(
  image("../../../../../results/plots/hyperparameter_tuning/experiments.svg", width: 100%),
  caption: [Training (solid) and validation (dashed) loss, and validation mAP50-95, for each tuning experiment.],
) <fig:tuning>
