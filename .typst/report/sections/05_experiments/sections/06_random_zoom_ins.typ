== Random Zoom-ins <sect:rand_zoom>

The final independent experiment investigated the use of random zoom-ins. This builds upon the increased-resolution experiment in @sect:res_increase, where improvements in small-object detection were observed. As discussed in @sect:rand_augs, random zoom-ins were applied with a probability of 50%, selecting a random region of the image with a maximum zoom of 30%. Bounding boxes extending beyond the selected region were clipped, while those that no longer intersected the region were removed.

The training parameters were as follows:

#figure(
  table(
    columns: 3,
    [*Image Size*], [*Batch Size*], [*Accum Steps*],
    [$768 times 768$], [8], [4]
  ),
  caption: [Training parameters for the random zoom-ins experiment.],
)


The results were as follows:

#figure(
  table(
    columns: 5,
    [*Epochs*], [*Best Epoch*], [*Best mAP50*], [*Best mAP50-95*], [*Best Small mAP50-95*],
    [18], [17], [0.721], [0.388], [0.243]
  ),
  caption: [Detection performance with random zoom-ins.],
)

In terms of overall performance, random zoom-ins improved the best mAP50-95 from 0.363 in @sect:res_increase to 0.388, representing an increase of 2.5 percentage points. This is similar to the improvement observed when introducing random horizontal flipping in @sect:rand_flips, suggesting that random spatial augmentations can be beneficial for overall detection performance.

However, the primary motivation for introducing random zoom-ins was to improve small-object detection. In this regard, the improvement was considerably smaller, with small-object mAP50-95 increasing from 0.242 to 0.243, a difference of just 0.1 percentage points. Such a small improvement could be attributed to variability in the training process rather than a meaningful improvement in small-object detection.

One potential explanation is that the zoom-ins were applied to randomly selected regions without considering the locations of annotated objects. Consequently, there was no guarantee that the selected region would contain a small object. A potential improvement would be to introduce object-aware zooming, where the selected region is influenced by the locations and density of bounding boxes. For example, the augmentation could target regions around the mean coordinates of annotated objects or bias the selection towards smaller bounding boxes, increasing the likelihood that small objects are enlarged during training.

Despite the limited improvement in small-object mAP50-95, a more significant improvement was observed in small-object recall. Compared with @sect:res_increase, small-object recall at IoU 0.50 increased from 0.787 to 0.850, an improvement of 6.3 percentage points. Similarly, recall at IoU 0.75 increased from 0.344 to 0.399, an improvement of 5.5 percentage points. This suggests that random zoom-ins improved the model's ability to detect small objects, even though the improvement in average precision was relatively small. However, the considerable difference between recall at IoU 0.50 and 0.75 indicates that precise localisation of small objects remains a challenge.

However, due to the improvement observed in overall detection performance, random zoom-ins were retained for the final training configuration.