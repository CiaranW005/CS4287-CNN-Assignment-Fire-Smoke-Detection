# CS4287 Assignment 1 — Convolutional Neural Networks

Build and evaluate a CNN in Python, investigate changes to its behaviour, and submit a PDF report and Jupyter notebook. This README converts the assignment brief into section-by-section tasks. Unchecked boxes describe work to complete, not verified project status.

## Assignment requirements

| Item | Requirement |
| --- | --- |
| Deadline | 23:59, Saturday 31 October 2026 (Week 8) |
| Weighting | 15% of the module |
| Language | Python |
| Deliverables | PDF report and Jupyter notebook |
| Implementation | Build the CNN layer by layer instead of using high-level framework model construction calls |

## Project direction and setup

The initial project idea is fire and smoke detection using a YOLO and ResNet hybrid architecture. This is a proposal, not a confirmed implementation.

- Dataset originally selected: [Smoke Fire Detection YOLO](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo).
- Existing setup instructions: [guide.md](guide.md).
- Existing folders: `data/`, `src/`, `notebooks/`, and `results/`.

## PDF report sections

Use the sections below in order. The marked sections total 15 marks. The statement of work, generative AI disclosure, and references are also Pass/Fail requirements.

### 1. Title page

**What needs to be done:** create the report cover page.

- [x] Include a title page.
- [x] Suggested contents: project title, module, assignment, team names, and student IDs.

### 2. Table of contents

**What needs to be done:** make the report easy to navigate.

- [x] Include a table of contents.
- [x] Check that headings and final page numbers match the report.

### 3. The dataset - 2 marks

**What needs to be done:** explain the data, visualise its important characteristics, and describe its preparation.

- [x] Identify and cite the dataset; explain the task, inputs, and target labels.
- [x] Visualise key attributes.
- [x] Discuss data correlation and feature engineering if applicable.
- [x] Explain preprocessing, including any normalisation.
- [x] Address the rubric's expectations: representativeness, inherent biases, data quality, cleaning, and augmentation where required.

### 4. Network structure and hyperparameters - 3 marks

**What needs to be done:** describe the chosen network, its configuration, and the reasons for its design.

- [x] Choose a suitable CNN architecture and explain the choice.
- [x] Include a network architecture diagram.
- [x] Implement the CNN layer by layer and describe its layers and connections.
- [x] Explain weight initialisation and activation functions.
- [x] Explain batch normalisation and regularisation, including whether and why they are used.
- [x] Record the hyperparameters used.
- [x] Describe transfer learning if used, including pretrained sources and what is frozen or trained.
- [x] Explain the architecture's distinguishing features. If using ResNet, include the required paragraphs on residual learning.

### 5. Loss function, optimiser, and cross-fold validation - 2 marks

**What needs to be done:** explain how learning works and how performance is validated.

- [ ] Define the cost/loss/error/objective function and justify its suitability.
- [ ] Discuss relevant alternative loss functions rather than naming a choice without explanation.
- [ ] Identify and explain the optimiser; justify the choice against alternatives.
- [x] Carry out and describe cross-fold validation, including the number of folds and data allocation.
- [x] Explain how models are compared and how a development set is used; these appear in the rubric's higher-band expectations.

### 6. Results and evaluation — 3 marks

**What needs to be done:** present measured results and explain what they reveal about the model.

- [ ] Include plots.
- [ ] Calculate accuracy and/or precision and/or recall, as appropriate to the task.
- [ ] Explain the numerical results and plots.
- [ ] Check for overfitting and underfitting.
- [ ] Connect performance to the model and hyperparameter choices.
- [ ] Calculate additional relevant metrics rather than only discussing them, as expected in the stronger rubric bands.

### 7. Independent experimental work — 3 marks

**What needs to be done:** run, analyse, and evaluate experiments on at least one of: data augmentation, architecture changes, hyperparameter changes, or deliberately engineered overfitting.

- [x] Define the experimental question and baseline.
- [x] Describe what changes between runs.
- [x] Train and evaluate the variants.
- [x] Present and compare results.
- [x] Explain why behaviour changed and what the team learned about neural computing.
- [x] Also address the overall objective of exploring the impact of varying hyperparameter(s).

### 8. Estimated level of difficulty — 2 marks

**What needs to be done:** write a very short paragraph explaining the team's opinion of the challenge, based on work actually completed.

| Example in the brief | Indicative marks |
| --- | --- |
| LeNet5 on MNIST | 0 |
| AlexNet on MNIST | 0.5 |
| AlexNet on MNIST with deliberately engineered overfitting | 1 |
| Inception, VGG, or ResNet | 1.5 |
| YOLO, Fast R-CNN, U-Net, or other | 2 |

- [x] Explain technical difficulty, original work, and layer-by-layer implementation where applicable.
- [x] Treat the table as guidance: the teaching team determines the actual difficulty mark.

### 9. Statement of work — Pass/Fail

**What needs to be done:** write one paragraph per team member describing their actual contribution.

- [x] Include all three members.
- [ ] Specify each person's implementation, experiments, analysis, and report work as applicable.
- [ ] Make contributions clear enough to support individual grading.

### 10. Use of generative AI — Pass/Fail

**What needs to be done:** disclose every prompt, summarise the response, and explain its use.

- [ ] List ALL prompts.
- [ ] Add one line summarising each subsequent response.
- [ ] Add another line explaining how it was used in the submission.
- [ ] Include this README-conversion interaction if it informs the submission.
- [ ] Distinguish sourced material from the team's own work.

**Restriction from the brief:** do not use generative AI to improve the English in the PDF. Preserve the students' voice and complete grammar and spelling checks before submission. The brief states that failure to comply results in an F grade.

Suggested log format:

| Complete prompt | One-line response summary | How it was used in the submission |
| --- | --- | --- |
| Paste prompt here | Summarise response | Describe actual use, or state not used |

### 11. References — Pass/Fail

**What needs to be done:** acknowledge all external material and show what the team added.

- [x] Cite the dataset, architecture sources, reused code, and other external material.
- [x] Follow the plagiarism-prevention guidance in UL's *Cite it Right*, referenced in the brief.
- [x] Clearly distinguish sourced content from new work.
- [x] Demonstrate deep understanding of reused projects, repository code, and LLM outputs.
- [x] Demonstrate added value when reusing material; the brief gives using a different dataset as an example.

## Jupyter notebook requirements

**What needs to be done:** submit working Python code with explanations demonstrating the team's understanding.

- [x] Use the filename specified in the brief: `CS4287-Assign2-ID1-ID2-ID3.ipynb`, replacing the IDs with team student numbers. See the ambiguity below.
- [x] First line: a comment containing team names and IDs.
- [x] Second line: a comment stating whether the code executes to the end without an error.
- [x] Third line: a comment linking to the third-party source if an existing implementation was reused.
- [ ] Have the team comment every critical line of code to demonstrate deep understanding. Comments may refer to PDF sections.
- [x] Verify execution to the end without an error.

## Final submission checklist

- [ ] PDF contains all required sections with readable figures, tables, and references.
- [ ] Notebook filename and opening comments match the confirmed requirements.
- [ ] Notebook executes fully and critical code has team-written explanations.
- [ ] Report results match the submitted code and experiments.
- [ ] Sources, reused material, AI prompts, and individual contributions are disclosed.
- [ ] Every member can explain the submission.
- [ ] Submit the PDF and notebook through Brightspace before the deadline.
