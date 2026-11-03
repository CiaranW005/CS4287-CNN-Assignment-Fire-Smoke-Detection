#pagebreak()

= Generative AI Use

*Ciaran Whelan (23370211)*

#figure(
  table(
    columns: (1fr, 1fr, 1.5fr),
    table.header(
      [*Complete prompt*],
      [*One-line response summary*],
      [*How it was used in the submission*],
    ),

    [How do I expand a dictionary stored in a single CSV cell?],
    [Explained how to convert a dictionary stored as a string into a Python dictionary.],
    [Used to process and extract evaluation metrics stored within CSV files for analysis and visualisation.],

    [Summarise the contributions of these papers to fire detection, highlight architectural differences, and return mAP50-95 results or equivalent metrics for comparison.],
    [Summarised the papers' architectures, reported detection performance],
    [Used to support the literature comparison section and identify architectural differences between our model and existing fire detection models.],

    [Generate BibTeX references for these websites: [...]],
    [Generated BibTeX entries containing publication information and source URLs.],
    [Used to create bibliography entries for documentation, tutorials and external code references consulted during development.],

    [How do I convert pretrained PyTorch weights to my own custom architecture?],
    [Explained how pretrained model weights can be loaded into compatible layers of a custom architecture.],
    [Used to guide the implementation of the pretrained ResNet backbone within our hybrid detection model.],

    [How do I calculate recall similarly to how MeanAveragePrecision is used?],
    [Explained how recall can be extracted from TorchMetrics evaluation results at different IoU thresholds and object sizes.],
    [Used to implement additional recall metrics for analysing detection performance across object sizes and IoU thresholds.],

    [Can you generate me a interactive buttons so click through images so I do not have 18 images displaying vertically in a notebook],
    [Explained how to create interactive image navigation buttons in a Jupyter notebook, allowing images to be displayed individually rather than all at once.],
    [Used to implement an interactive image viewer for examining invalid dataset samples, reducing notebook clutter while allowing users to navigate between images.]
  ),
  caption: [Ciaran's generative AI prompts, responses and how they were used.],
) 

#pagebreak()

*Emma O'Reilly (23304472)*

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

#pagebreak()
*Hersh Singh (23371137)*

I used GenAI for my parts of the project. These are the main prompts I used; a full log of every prompt is available on request.

#figure(
  table(
    columns: (1fr, 1.0fr, 1.5fr),
    align: (left, left, left),
    [*Complete prompt*], [*One-line response summary*], [*How it was used in the submission*],
    ["Integrate everyones code the best we can as a draft for the final Notebook, but its vital if we have the option to use preexisting csvs as results to save computational time"], [Built a draft of the final notebook with everyone's code, which loads saved results and only trains what is missing.], [As a draft for the final notebook which I then checked and edited.],
    ["Before we do that double check that the outputs ran match the report"], [Compared the notebook outputs with the numbers in the report.], [Confirm report matches the notebook.],
    ["Lets clean up the markdown for the final notebook draft too, let's remove anything unneccessary and keep it consise"], [Shortened the text cells of the notebook.], [In the final notebook.],
    ["Is there anything that may appear as vague that I should comment in my segments of the final notebook code? Highlight these as \[edit\] so I can return to"], [Listed lines in my code that needed comments marked \[edit\].], [I wrote the comments for lines labelled \[edit\].],
    ["Ok, cool, can you update any of my other notebooks with the exact same comments as well then"], [Copied my comments into my cross-validation and hyperparameter tuning notebooks.], [In the notebooks mentioned.],
    ["Can you check my near-duplicates paragraph against the results and flag discrepencies??"], [Checked each number in the paragraph and table against the results and flagged two rounding errors.], [Correct the two rounded numbers in my near-duplicates table (@tab:test_subsets): mAP50 0.860 to 0.859 and mAR100 0.534 to 0.533.],
  ),
  caption: [Hersh's generative AI prompts, responses and how they were used.],
) <tab:genai_use_hersh>
