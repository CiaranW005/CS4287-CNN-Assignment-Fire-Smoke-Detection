== Baseline Run

The baseline run was conducted to establish an initial performance benchmark for the model, allowing subsequent experiments to be compared against it. The model was trained using the following parameters:

#figure(
  table(
    columns: 3,
  
    [*Image Size*], [*Batch Size*], [*Accum Steps*],
    [$640 times 640$], [14], [2]
  ),
  caption: [Training parameters for the baseline experiment.],
)

After 10 epochs, the model achieved the following results:

#figure(
  table(
    columns: 3,
    [*Epochs*], [*Best mAP50*], [*Best mAP50-95*],
    [10], [0.657], [0.335]
  ),
  caption: [Detection performance after 10 epochs of baseline training.],
)

These results give an overview of the initial performance of the model, with an mAP50 of 0.657 indicating reasonable detection performance at an IoU threshold of 0.5. However, the lower mAP50-95 of 0.335 suggests that performance decreases at stricter IoU thresholds, potentially indicating difficulties with precise bounding-box localisation. Further experimentation will examine how performance changes across these thresholds. As mentioned in @sec:model_compare, recall will also be used to evaluate detection performance across different bounding-box sizes, particularly smaller annotations.