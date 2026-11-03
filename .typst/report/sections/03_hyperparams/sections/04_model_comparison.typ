// What metrics we used to make decisiosn (map95 as overall, recall to assess model quality on smaller boxes)

// TODO explain what IoU is
== Model Comparison <sec:model_compare>

All models are scored with COCO average precision from torchmetrics @torchmetrics_map. Before scoring, overlapping detections are removed with non-maximum suppression, using the settings Ultralytics validates with (confidence 0.001, IoU 0.7, at most 300 boxes per image).

#figure(
  table(
    columns: 2,
    align: (left, left),
    [*Metric*], [*Meaning*],
    [mAP50-95], [Average precision over IoU thresholds 0.50 to 0.95, averaged between both classes],
    [mAP50], [Average precision at IoU 0.50],
    [recall50], [Share of boxes found at IoU 0.50],
    [mAR100], [Recall averaged over IoU 0.50 to 0.95, up to 100 detections per image],
    [Small / medium / large], [Same metrics for boxes under $32^2$, between $32^2$ and $96^2$, and over $96^2$ pixels],
  ),
  caption: [Metrics used to compare models.],
) <tab:metrics>

mAP50-95 is our main metric, as it rewards boxes that are both correct and tight. It is used to pick the best epoch of each run and to compare runs. For a fire detector, the consequences of missing a fire are far greater than a false alarm, so we also check recall, including for small boxes, as most fire boxes are small.

During tuning, every run uses the same seed and the same images, so only the setting being tested changes. We treat differences under 0.01 mAP50-95 as noise and settings that beat the base by more than that are cross-validated.
