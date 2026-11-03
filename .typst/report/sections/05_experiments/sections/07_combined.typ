== Final Training Run

This final training run combines the configurations investigated in the previous experiments. While not all experiments were strictly independent, certain changes, such as the learning-rate plateau reduction in @sect:lr_plateau and the increased image resolution in @sect:res_increase, demonstrated sufficient improvements to be retained in subsequent experiments. This final run investigates how combining random horizontal flips from @sect:rand_flips, cosine annealing from @sect:cos_anneal, and random zoom-ins from @sect:rand_zoom affects model performance. The resulting configuration will serve as the baseline for subsequent hyperparameter tuning.

The parameters of this run were as follows:

#figure(
  table(
    columns: 3,
    [*Image Size*], [*Batch Size*], [*Accum Steps*],
    [$768 times 768$], [8], [4]
  ),
  caption: [Training parameters for the final experiment.],
)

The results were as follows:

#figure(
  table(
    columns: 5,
    [*Epochs*], [*Best Epoch*], [*Best mAP50*], [*Best mAP50-95*], [*Best Small mAP50-95*],
    [27], [27], [0.733], [0.412], [0.266]
  ),
  caption: [Detection performance with the final configuration.],
)

This was the best-performing training run, achieving an mAP50-95 of 0.412 and exceeding 0.400 for the first time across the experiments. This represents a considerable improvement in overall detection performance compared with the previous configurations. One potential explanation is that combining random horizontal flips and random zoom-ins introduced greater variation into the training images than either augmentation individually. This increased variation may have allowed the model to generalise better to different orientations, positions and scales of fire and smoke objects.