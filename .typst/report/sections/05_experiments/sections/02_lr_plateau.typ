== LR Plateau Reductions <sect:lr_plateau>

Due to our model starting from a pretrained backbone with a reduced learning rate, as mentioned in @sect:transfer_learn, the initial learning rate may not remain suitable throughout training. After several epochs, once the randomly initialised YOLO neck and head have begun learning, a smaller learning rate may help the model converge towards a better local minimum.

To identify when the learning rate should be reduced, a custom plateau detector was implemented using linear regression over a moving window of the five most recent validation losses. The slope is normalised by the absolute mean loss of the window, calculated as $m_"relative" = m / (|mu_L| + epsilon)$. A plateau is detected when the relative slope exceeds $-0.001$, indicating that the validation loss is improving by less than approximately 0.1% per epoch or has begun increasing. The learning rate is then halved across all parameter groups in the optimiser.

This approach was preferred over a conventional fixed-delta detector because regression examines the overall trend across several epochs rather than individual improvements. Furthermore, normalising the slope makes the threshold less dependent on the magnitude of the validation loss, allowing the same criterion to be applied across different loss scales without requiring as much manual adjustment.

The parameters of this training run were as follows:

#figure(
  table(
  columns: 3,
  [*Image Size*], [*Batch Size*], [*Accum Steps*],
  [$640 times 640$], [14], [2]
  ),
  caption: [Training parameters for the learning-rate plateau experiment.],
)

The effects of the plateau scheduler were examined by comparing the best performance achieved before and after the learning-rate reduction. Epoch 10 was the final epoch before the reduction, while epoch 11 was the first epoch evaluated after it.

#figure(
  table(
  columns: 3,
  [*Epoch*], [*Best mAP50*], [*Best mAP50-95*],
  [10], [0.669], [0.342],
  [11], [0.688], [0.363]
  ),
  caption: [Best detection performance before and after the learning-rate reduction.],
)

The model achieved its best metrics at epoch 11, immediately following the learning-rate reduction. This represented an improvement of 0.019 in mAP50 and 0.021 in mAP50-95 over the previous best results. However, no further improvements were achieved during the remaining training, suggesting that while the initial plateau reduction was beneficial, its effectiveness was limited beyond this point.

Due to the improvements observed following the plateau reduction, this approach was retained for subsequent experiments and used as the new baseline for comparison. Later experiments will investigate whether alternative learning-rate scheduling strategies can improve the model's performance further.