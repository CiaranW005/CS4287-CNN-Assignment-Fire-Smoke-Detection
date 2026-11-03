== ResNet-50 Backbone

The backbone is ResNet-50, implemented layer-by-layer in PyTorch. It's made up of a stem followed by four stages of residual blocks. We drop the average pooling at the end, since we only need the backbone to extract features, not to classify images.

The image goes through a $7 times 7$ convolution with stride 2, batch normalisation and ReLU, and then a $3 times 3$ max-pool with stride 2. By the time it reaches the residual stage, the image is 4 times smaller.


For the residual blocks, each has three convolutions; a $1 times 1$ convolution that squeezes the channels down to a width $w$, a $2 times 3$ convolution at that width, and another $1 times 1$ convolution that brings the image back to $4w$ channels. Every convolution is followed by batch normalisation, with the first two also having ReLU. The output of the block is $"ReLU"(F(x) + S(x))$, where $S$ is the shortcut. If the input and output shapes match, the shoertcut is just the identity. If they don't, we use a $1 times 1$ convolution with batch normalisation to make them match.

The four stages have 3, 4, 6 and 3 blocks. The first block in each stage uses a projection shortcut to change the number of channels. In stages 2-4 it also halves the resolution.

#figure(
  table(
    columns: 4,
    [*Layer*], [*Blocks*], [*Output ($C times H times W$)*], [*Stride*],
    [Input], [--], [$3 times 768 times 768$], [1],
    [Stem], [--], [$64 times 192 times 192$], [4],
    [Stage 1], [3], [$256 times 192 times 192$], [4],
    [Stage 2 ($C_3$)], [4], [$512 times 96 times 96$], [8],
    [Stage 3 ($C_4$)], [6], [$1024 times 48 times 48$], [16],
    [Stage 4 ($C_5$)], [3], [$2048 times 24 times 24$], [32],
  ),
  caption: [Feature map shapes through the ResNet-50 backbone for a $768 times 768$ input.],
) 

We take the outputs of stages 2, 3 and 4 and pass them on to the neck. These are strides 8, 16 and 32, which line up with the 3 detection scale sin YOLOv8. The finer maps help with small objects like distant fires, while the coarser ones are better for large patches of smoke.

