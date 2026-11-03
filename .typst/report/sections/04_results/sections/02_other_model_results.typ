== Results from Other Models

This section compares our model's performance against results reported in other research using the same underlying dataset or closely related subsets. The dataset used in this project was obtained from the Smoke-Fire-Detection-YOLO dataset on Kaggle, which is also distributed as a version of the D-Fire dataset.

Two papers were selected for comparison:

- *Cross-Dataset Evaluation of YOLOv8 for Unmanned Aerial Vehicle Fire and Smoke Detection* @ruangsang2026yolov8, which evaluates multiple YOLOv8 architectures on the D-Fire dataset.
- *TFNet: Transformer-Based Multi-Scale Feature Fusion Forest Fire Image Detection Network* @liu2025tfnet, which investigates a specialised architecture combining a modified ResNet backbone, multi-scale feature fusion and transformer-based detection.

These studies were selected to compare our custom detection architecture against both established YOLO models and an alternative architecture designed specifically for fire and smoke detection. Differences in training data, dataset splits and experimental configurations are also considered when discussing potential reasons for their higher performance.

=== YOLOv8 Cross-Dataset Evaluation

The first paper, @ruangsang2026yolov8, evaluates three standard YOLOv8 architectures: YOLOv8n, YOLOv8s and YOLOv8m. Unlike our model, which uses an ImageNet-pretrained ResNet backbone with a randomly initialised YOLO-style neck and detection head, these models were initialised using COCO-pretrained object detection checkpoints. Consequently, their backbones, necks and detection heads had already learned features relevant to object detection before being fine-tuned on fire and smoke images.

The YOLOv8 architectures also use established multi-scale feature extraction and fusion to combine information from different spatial resolutions. This is particularly relevant to our model, as the results in @tab:test_size identified small-object localisation as an area requiring further improvement.

The paper reports the following results on the D-Fire test set:

#figure(
table(
columns: 3,
table.header([Model], [mAP50], [mAP50-95]),
[Our Model], [0.7414], [0.4070],
[YOLOv8n], [0.7765], [0.4489],
[YOLOv8s], [0.7830], [0.4524],
[YOLOv8m], [0.7904], [0.4638]
),
caption: [Comparison between our model and the YOLOv8 results reported by @ruangsang2026yolov8.],
)

All three YOLOv8 variants achieved higher mAP50 and mAP50-95 scores than our model. Interestingly, the difference between the smallest and largest YOLOv8 variants was relatively small, with mAP50-95 increasing from 0.4489 to 0.4638. This suggests that increasing model capacity alone did not produce substantial improvements in detection performance.

One potential explanation for the higher scores is the difference in pretraining. Our model required the detection neck and head to learn from random initialisation, whereas the YOLOv8 models benefited from pretrained object detection weights. Furthermore, the paper used a training configuration of up to 100 epochs with a broader augmentation pipeline, including mosaic augmentation, compared with our shorter training runs.

However, the study used the official D-Fire training and testing partitions, while our project used a different selection of images and dataset split. Therefore, the results provide a useful comparison of model performance but cannot establish that the architectural differences alone caused the higher scores.

=== TFNet: Transformer-Based Multi-Scale Feature Fusion

The second paper, @liu2025tfnet, proposes TFNet, a specialised fire and smoke detection architecture based on Real-Time Detection Transformer(RT-DETR). Unlike our model's conventional convolutional backbone and YOLO-style detection architecture, TFNet introduces several components designed to improve feature extraction, multi-scale feature fusion and bounding-box localisation.

Their first modification is the SRModule, which replaces parts of a ResNet18 backbone with specialised convolutional operations to extract a more diverse range of features. The second is the CG-MSFF Encoder, which combines adaptive feature fusion with attention mechanisms to better integrate information from different feature-map resolutions. Finally, TFNet uses a transformer-based decoder to refine object predictions and Wise-IoU loss to improve bounding-box regression.

#figure(
table(
columns: 3,
table.header([Model], [mAP50], [mAP50-95]),
[Our Model], [0.7414], [0.4070],
[TFNet], [0.8120], [0.4680]
),
caption: [Comparison between our model and TFNet, as reported by @liu2025tfnet.],
)

The higher performance of TFNet could be partly attributed to its specialised feature-fusion architecture. In particular, the attention-based combination of features at different resolutions may help preserve information about small fire and smoke objects, which our model struggled to localise accurately. Additionally, Wise-IoU modifies the contribution of bounding-box regression errors during training, potentially improving localisation performance.

However, TFNet was trained on a selected subset of 9,869 forest-related images from D-Fire, using a 70/20/10 split and 200 training epochs. These differences in dataset composition and training methodology mean that its results are not directly equivalent to those obtained by our model.

Overall, both papers demonstrate that established object detection architectures can achieve higher performance on related fire and smoke detection tasks. The YOLOv8 results highlight the potential benefits of detection-specific pretraining and established multi-scale architectures, while TFNet demonstrates how specialised feature fusion and bounding-box regression methods can further improve detection performance. These approaches provide potential directions for improving our model, particularly its performance on smaller objects.