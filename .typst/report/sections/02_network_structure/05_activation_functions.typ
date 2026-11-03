== Activation Functions

Our model used both ReLU and SiLU as activation functions. The ResNet backbone makes use of the ReLU while the Adapter, neck and head all use SiLU. The output of the head uses the sigmoid activation function for classification and softmax for the box regression values.

=== ReLU (backbone)
The ReLU activation is used after the stem convolution, after the first two convolutions of each bottleneck and after the residual addition.
The ReLU activation function is defined as: $ "ReLU"(x) = max(0, x) = cases(x & "if" x > 0, 0 & "if" x <= 0) $
Our decision to use the ReLU activation function was based on the fact that the ResNet architecture was designed and pretrained using ReLU, along with our model's initialised ImageNet weights being learned under ReLU.

The main drawback of using ReLU is the problem known as "dying ReLU". The gradient of ReLU is 0 for negative inputs, so if a unit's input becomes negative for every image, it stops updating and always outputs 0. We use SiLU in the adapter, neck and head, so this problem does not affect the new layers.

=== SiLU (adapter, neck and head)
The SiLU activation function is used in the neck and head of our model. The SiLU function is defined as:
$ "SiLU"(x) = x dot sigma(x) = x / (1 + e^(-x)) $
SiLU mitigates the "dying ReLU" problem by allowing for non-zero outputs for negative inputs, which helps to maintain gradient flow during training. 

=== Sigmoid (head)
The sigmoid activation function is used in the head of our model for classification. The sigmoid function is defined as:
$ "sigmoid"(x) = 1 / (1 + e^(-x)) $
The sigmoid function outputs values between 0 and 1, which is suitable for binary classification tasks. In our case, it is applied to each class score separately, giving an independent probability that a predicted box contains fire and that it contains smoke.

=== Softmax (head)
The softmax activation function is used in the head of our model for box regression. The softmax function is defined as:
$ "softmax"(x_i) = e^(x_i) / (sum_(j=1)^16 e^(x_j)) $
Softmax turns a set of scores into probabilities that add up to 1. For each edge of a box, the head gives a score to 16 possible distances from the centre. Softmax turns these into probabilities, and the predicted distance is their weighted average.

