== Loss Function <sec:loss>

Object detection is a multi-task problem. For every object, the model has to predict where it is (a bounding box) and what it is (fire or smoke). No single standard loss covers both, so we imported the YOLOv8 detection loss @ultralytics_yolov8, which is a weighted sum of three terms:
$ L = lambda_"box" L_"CIoU" + lambda_"cls" L_"BCE" + lambda_"dfl" L_"DFL" $

Before the loss is calculated, each ground-truth box is matched to the predictions that best fit it, using the Task-Aligned Assigner. All other predictions are treated as background, so images with no boxes still teach the model what background looks like.

The three terms are:
- *CIoU box loss*: penalises low overlap between the predicted and true box, distance between their centres, and different aspect ratios. Overlap is what mAP uses to judge a detection, and the aspect ratio term helps with smoke, whose boxes vary a lot in shape (@fig:smoke_box_props).
- *Binary cross-entropy (BCE) classification loss:* scores each class separately, so a region can be both fire and smoke, which often happens in our images.
- *Distribution Focal Loss (DFL)*: instead of predicting one exact position for each box edge, the model predicts a range of likely positions. This suits smoke, whose edges are blurry.

The gains $lambda$ balance the three terms. The box term has the highest gain because localisation is the harder part of the task. We used the Ultralytics defaults (@tab:hyperparameters) for gains values.

=== Alternative Loss Functions
We considered alternative loss functions, but opted not to use them for the following reasons:
- *L1 / L2 on box coordinates*: depend on box size and do not directly improve overlap.
- *IoU, GIoU and DIoU:* IoU gives no gradient when boxes do not overlap; GIoU and DIoU fix this, but lack CIoU's aspect ratio term.
- *Softmax cross-entropy:* treats fire and smoke as mutually exclusive.
- *Focal loss*: handles the large number of background predictions, but the Task-Aligned Assigner already reduces this problem.

