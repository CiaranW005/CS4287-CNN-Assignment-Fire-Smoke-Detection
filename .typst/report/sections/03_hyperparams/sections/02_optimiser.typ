== Optimiser <sec:optimiser>

We used the AdamW optimiser @pytorch_adamw. It keeps a running average of recent gradients and of their squares, and uses these to give every weight its own step size. Weights with large, noisy gradients take smaller steps, while weights with small gradients take larger ones.

Our model combines a pretrained backbone with newly initialised layers, and the loss combines three terms with different gains, so gradient sizes vary a lot across the network. AdamW's per-weight step sizes handle this without us having to tune each part separately. It also converges quickly with little tuning, which was important because each configuration had to be trained five times for cross-validation.

The learning rates and weight decay used with AdamW were chosen through hyperparameter tuning and are listed in @tab:hyperparameters.

=== Alternative Optimisers
Some other optimisers we considered were:
- *SGD with momentum:* commonly used to train ResNet and YOLO, but is more sensitive to the learning rate and usually needs longer training.
- *Adam:* weight decay is weakened by the adaptive step size, which would give less reliable regularisation.
- *RMSProp:* adapts step sizes like Adam but has no momentum, so updates would be noisier.
