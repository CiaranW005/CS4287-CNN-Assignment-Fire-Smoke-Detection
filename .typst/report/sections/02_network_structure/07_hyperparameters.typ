== Hyperparameters <sec:hyperparameters>

The main hyperparameters of our final model are listed in @tab:hyperparameters. The learning rate, weight decay and backbone learning rate were chosen through hyperparameter tuning (@tab:tuning), while the loss gains are the Ultralytics defaults @ultralytics_yolov8.

#figure(
  table(
    columns: 2,
    align: (left, left),
    [*Hyperparameter*], [*Value*],
    [Input size], [$768 times 768$],
    [Batch size], [8, with 4 gradient accumulation steps (32 images per update)],
    [Epochs], [Up to 50 (early stopping ended training after 27)],
    [Optimiser], [AdamW],
    [Learning rate $eta$], [$10^(-3)$],
    [Backbone learning rate], [$0.1 times eta$],
    [Weight decay $lambda$], [$10^(-4)$],
    [Learning rate schedule], [Halved on the first validation loss plateau, then cosine decay to $0.1 times eta/2$; training stops on the second plateau],
    [Loss gains], [Box 7.5, classification 0.5, DFL 1.5],
  ),
  caption: [Main hyperparameters of the final model.],
) <tab:hyperparameters>
