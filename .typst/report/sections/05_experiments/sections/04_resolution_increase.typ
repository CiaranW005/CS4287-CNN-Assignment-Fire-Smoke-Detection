== Increasing the Resolution <sect:res_increase>

There is a concern regarding the performance of small bounding boxes in the dataset. As observed in @sect:rand_flips, small objects achieved significantly worse performance compared with the overall average. This experiment investigates whether increasing the image resolution can improve the detection of these smaller objects.

The parameters for this run were as follows:

#figure(
  table(
    columns: 3,
    [*Image Size*], [*Batch Size*], [*Accum Steps*],
    [$768 times 768$], [8], [4]
  ),
  caption: [Training parameters for the increased-resolution experiment.],
)

The results were as follows:

#figure(
  table(
    columns: 5,
    [*Epochs*], [*Best Epoch*], [*Best mAP50*], [*Best mAP50-95*], [*Best Small mAP50-95*],
    [17], [14], [0.687], [0.363], [0.242]
  ),
  caption: [Detection performance with an input resolution of $768 times 768$.],
)

Although random horizontal flipping was removed from this experiment, the increased resolution still achieved a higher small-object mAP50-95 of 0.242 compared with 0.227 in @sect:rand_flips. This is particularly interesting given that horizontal flipping had previously improved overall model performance. While the experiment does not isolate the effect of resolution, the improvement in small-object detection provides justification for continuing with the higher resolution in subsequent experiments.

Recall was also recorded at different IoU thresholds to investigate how detection performance changes as the required bounding-box overlap becomes stricter. At epoch 14, small-object recall decreased from 0.784 at IoU 0.50 to 0.325 at IoU 0.75, indicating that precise localisation of small objects remains a significant challenge.