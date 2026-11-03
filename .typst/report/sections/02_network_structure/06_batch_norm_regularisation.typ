== Batch Normalisation <sec:batch_norm>
Batch normalisation is used heavily throughout our hybrid model, after almost every convolutional layer at all stages in the model. Batch normalisation is used to normalise the inputs to each layer to a mean of 0 and a standard deviation of 1. This helps to stabilise the training process and allows for higher learning rates, which can lead to faster convergence.

In the ResNet backbone, we use batch normalisation after the stem convolution, after all three convolutions of each bottleneck and after the $1 times 1$ projection shortcut convolution. It is not used after the residual addition, which is only followed by ReLU. In the YOLOv8 neck, we use batch normalisation after every convolutional layer. In the YOLOv8 head, we use batch normalisation after every convolutional layer except for the final output layer, as it outputs the final predictions.

== Regularisation <sec:regularisation>
Rather than a single mechanism, our model is regularised through a combination of techniques, most of which are described in more detail in other sections:

- *Data augmentation:* the colour jitter (@sec:colour_jitter), horizontal flips and random zoom-ins (@sect:rand_augs) are applied only to the training set and are re-sampled every epoch, so the model rarely sees the same image twice.
- *Weight decay:* AdamW @pytorch_adamw applies a weight decay of $lambda = 10^(-4)$ to all parameters, shrinking the weights slightly at each step and penalising large weights.
- *Differential learning rate:* training the backbone at $0.1 times eta$ keeps it close to its pretrained ImageNet weights (@sec:transfer_learning), limiting how much it can overfit our much smaller dataset.
- *Early stopping:* a plateau is detected when a line fitted to the last 5 validation losses is no longer falling by more than 0.1% per epoch. On the first plateau, the learning rate is halved and then decayed with a cosine schedule, and training stops on the second plateau. The weights from the epoch with the best validation mAP50-95 are kept, so later overfitted epochs are never used.
- *Batch normalisation:* as the statistics in @sec:batch_norm are computed per mini-batch, they add a small amount of noise to the activations during training, which has a mild regularising effect.

