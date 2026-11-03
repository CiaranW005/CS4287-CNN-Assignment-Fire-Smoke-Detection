= Generative AI Use

Generative AI tools were used sparsely throughout the project, primarily for explanations, troubleshooting and debugging. 
The list below outlines all prompts used, along with the responses generated:

#figure(
  table(
    columns: (1fr, 1.0fr, 1.5fr),
    align: (left, left, left),
    [*Complete prompt*], [*One-line response summary*], [*How it was used in the submission*],
    ["How does a yolo model built layer by layer differ in code to importing the yolo model for use in the hybrid model?"], [The response listed the differences, with code examples.], [Helped me estimate the difference in difficulty between encorporating an imported model with a predefined backbone and building the model layer by layer - I ultimately decided to build the neck of the model layer-by-layer while importing the YOLO Detect head along with the loss.], ["Explain FPN top-down and PAN botom-up in a simple, general context"],[The response summarised how the concepts work - with upsampling, downsampling and concatenation, making reference to the research papers that first introduced them.],[Helped me understand the logic behind the neck of our hybrid model, and how it can be represented in a diagram.], ["What is the SiLU function defined as?"], [The exact definition of the function, in mathematical terms and casual language.], [Helped me understand why the activation function was appropriate for use in the YOLOv8 model, and how it differs from the ReLU function used in the ResNet backbone.], ["How can I add mAP to evaluation metrics?"], [Outlined the steps involved with adding the extra metric - the necessary libraries, a type for storing the result, the file the computation should be completed in how to show it in the epoch log.], [Used to help implementation of the metric in a structured way, and to ensure the metric was being computed correctly.]
  ),
  caption: [Generative AI prompts, responses and how they were used.],
) <tab:genai_use>
