== Residuals

Residuals appear in our ResNet-50 backbone, where each of the 16 bottleneck blocks has one residual connection. These connections make a deep network like ResNet-50 easier to train, because they help avoid the vanishing gradient problem @he2016resnet.

=== The Vanishing Gradient Problem
During backpropagation, the gradient is passed backwards through the network one layer at a time, and is multiplied by each layer's weights along the way. In a deep network, these repeated multiplications can shrink the gradient to almost zero before it reaches the early layers. Those layers then barely update, and training slows down or stalls.

=== How Residual Learning Helps
In a normal block, the layers have to learn the full output $cal(H)(x)$ from the input $x$. In a residual block, the layers only learn the change that needs to be made to the input, $cal(F)(x) = cal(H)(x) - x$. The input is then added back on using a shortcut connection:
$ y = "ReLU"(cal(F)(x) + x) $
This is easier to learn, as each block only has to make a small change to the features it receives. If a block is not needed, its layers can just learn to output zero, and the input passes through unchanged.

The shortcut also helps with vanishing gradients. During backpropagation, the gradient can travel back through the shortcut without being multiplied by any weights. So even if the gradient through the layers becomes very small, the early layers still receive a useful gradient.

=== Residuals in Our Model
In our bottleneck blocks, the layers that learn $cal(F)(x)$ are the three convolutions and their batch normalisations (@sec:resnet). The input is added after the last batch normalisation, and the result is passed through ReLU. There are two types of shortcut:

- *Identity shortcut:* if the input and output are the same shape, the input is added on directly. This is used in 12 of the 16 blocks.
- *Projection shortcut:* the first block of each stage changes the number of channels, and in stages 2-4 also halves the image size. The input can no longer be added directly, so a $1 times 1$ convolution and batch normalisation first reshape it to match. This is used in the other 4 blocks.

Our ImageNet pretrained weights were trained with these shortcuts, so we kept them in our model to make the most of transfer learning (@sec:transfer_learning).
