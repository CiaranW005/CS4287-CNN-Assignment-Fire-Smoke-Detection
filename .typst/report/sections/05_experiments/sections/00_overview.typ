= Independent Experiments

This section examines how different augmentations, image resolutions and regularisation techniques affect model performance during training. The aim is to identify which configurations improve performance and should be carried forward into the final experiments discussed in @sect:hyperparams.

Hyperparameters such as learning rate and weight decay are kept constant across experiments unless they are the specific variable being investigated. This helps maintain consistency between training runs, allowing the effects of individual changes to be compared.

Results across experiments are compared using only the best values achieved on the validation set. Comparing metrics from every epoch would considerably expand the discussion without necessarily providing additional insight beyond the results already presented.