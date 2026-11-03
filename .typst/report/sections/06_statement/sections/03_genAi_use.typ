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

*Hersh Singh (23371137)*

I used GenAI for my parts of the project. These are the main prompts I used; a full log of every prompt is available on request.

+ "Integrate everyones code the best we can as a draft for the final Notebook, but its vital if we have the option to use preexisting csvs as results to save computational time" \
  _Response:_ Built a draft of the final notebook with everyone's code, which loads saved results and only trains what is missing. \
  _Used:_ As a draft for the final notebook which I then checked and edited.

+ "Before we do that double check that the outputs ran match the report" \
  _Response:_ Compared the notebook outputs with the numbers in the report. \
  _Used:_ Confirm report matches the notebook.

+ "Lets clean up the markdown for the final notebook draft too, let's remove anything unneccessary and keep it consise" \
  _Response:_ Shortened the text cells of the notebook. \
  _Used:_ In the final notebook.

+ "Is there anything that may appear as vague that I should comment in my segments of the final notebook code? Highlight these as \[edit\] so I can return to" \
  _Response:_ Listed lines in my code that needed comments marked \[edit\]. \
  _Used:_ I wrote the comments for lines labelled \[edit\].

+ "Ok, cool, can you update any of my other notebooks with the exact same comments as well then" \
  _Response:_ Copied my comments into my cross-validation and hyperparameter tuning notebooks. \
  _Used:_ In the notebooks above.

+ "Can you check my near-duplicates paragraph against the results and flag discrepencies??" \
  _Response:_ Checked each number in the paragraph and table against the results and flagged two rounding errors. \
  _Used:_ Correct the two rounded numbers in my near-duplicates table (@tab:test_subsets): mAP50 0.860 to 0.859 and mAR100 0.534 to 0.533.
