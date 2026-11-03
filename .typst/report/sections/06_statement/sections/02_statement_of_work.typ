= Statement of Work

*Ciaran Whelan (23370211)*

*Emma O'Reilly (23304472)*

I implemented the YOLOv8 neck and head and combined these with the ResNet backbone to create our hybrid model. I implemented the adapter to ensure the outputs of our ResNet backbone match the inputs expected by the YOLO neck. I implemented the neck of the model layer-by-layer, including the top-down FPN and bottom-up PAN passes. I integrated the Ultralytics v8DetectionLoss function into our model and added the mAP50 metric for evaluation. To ensure we could use the imported YOLO loss, I rewrote the dataset to return labels as tensors. I also contributed to the logic for gradient accumulation and added the differential learning rate for the backbone to ensure the hybrid model doesn't lose the benefits of the pretrained ResNet backbone. I added logging per epoch which saves losses and mAP50 to a CSV file for each training epoch. For the report, I wrote the Network Structure section.

*Hersh Singh (23371137)*

I created the hyperparameter tuning and cross-validation notebooks. I grouped near-duplicate images with a difference hash for cross-validation, so that no group is split across two folds. I measured how many validation and test images have a near-duplicate in the train split. I ran the nine tuning experiments and both five-fold cross-validation runs on a GPU, and set up my notebooks so that a failed run is retried when a run is resumed (in case of power failure, crashes etc.). I also implemented the final submission notebook, including the test evaluation with and without near-duplicates. In the report, I wrote the Cross-Validation, Model Comparison and Hyperparameter Tuning sections, and the near-duplicate analysis in Results.
