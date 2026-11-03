== Cosine Annealing <sect:cos_anneal>

As mentioned at the end of the discussion in @sect:lr_plateau, it was hypothesised that introducing a learning-rate scheduler could further improve model performance following the initial plateau reduction. By gradually decreasing the learning rate over subsequent epochs, the scheduler may allow the model to make smaller parameter updates and stabilise training during its final stages.

For example, examining the final epochs of the increased-resolution experiment in @sect:res_increase, @tab:final_map50-95 shows that following the learning-rate reduction at epoch 13, subsequent epochs produced only marginal improvements or decreases in mAP50-95. This motivated investigating whether further reductions in the learning rate through cosine annealing could improve the model's performance.

#let results = csv("../../../../../results/models/20261029_094103/history.csv")

#figure(
  table(
    columns: 2,
    [*Epoch*], [*mAP50-95*],
    ..results.slice(13).map(row => (
      [#row.at(0)],
      [#calc.round(float(row.at(4)), digits: 3)],
    )).flatten()
  ),
  caption: [Validation mAP50-95 during the final epochs of the increased-resolution experiment.],
) <tab:final_map50-95>

The parameters of this training run were as follows:

#figure(
  table(
    columns: 3,
    [*Image Size*], [*Batch Size*], [*Accum Steps*],
    [$768 times 768$], [8], [4]
  ),
  caption: [Training parameters for the cosine annealing experiment.],
)

Results shown as:

#figure(
  table(
    columns: 5,
    [*Epochs*], [*Best Epoch*], [*Best mAP50*], [*Best mAP50-95*], [*Best Small mAP50-95*],
    [15], [15], [0.700], [0.373], [0.242]
  ),
  caption: [Detection performance with Cosine Annealing.],
)

Cosine annealing was selected as the learning-rate scheduler, although exponential decay was also considered. Since the training process already incorporates a plateau-based learning-rate reduction, cosine annealing was preferred because it allows the learning rate to decrease smoothly following the initial reduction. The plateau detector determines when to begin reducing the learning rate, while cosine annealing gradually decreases it over the remaining epochs, allowing smaller parameter updates as training progresses.

The scheduler is activated when the first plateau is detected, at which point the learning rate is halved across all optimiser parameter groups. Cosine annealing is then applied over subsequent training epochs, gradually decreasing the learning rate to a tenth of the current learning rate over 8 epochs, according to a cosine-shaped curve. This allows the model to make progressively smaller parameter updates as training continues. The scheduler remains active until training terminates, either through early stopping or reaching the maximum number of epochs.

Comparing the results with the increased-resolution experiment in @sect:res_increase, introducing cosine annealing improved the best mAP50-95 from 0.363 to 0.373, representing an increase of 1 percentage point. The best mAP50 also increased from 0.687 to 0.700, while small-object mAP50-95 remained approximately unchanged at 0.242. This suggests that the improvement was primarily in overall detection performance rather than small-object detection.

The model achieved its best overall mAP50-95 at epoch 15, compared with epoch 14 in the previous experiment. These results support the hypothesis that further reducing the learning rate after the initial plateau reduction can improve model performance. Consequently, cosine annealing was retained for the final experiment.