= Statement of Work

*Ciaran Whelan (23370211)*

*Emma O'Reilly (23304472)*

*Hersh Singh (23371137)*

I created the hyperparameter tuning and cross-validation notebooks. I grouped near-duplicate images with a difference hash for cross-validation, so that no group is split across two folds. I measured how many validation and test images have a near-duplicate in the train split. I ran the nine tuning experiments and both five-fold cross-validation runs on a GPU, and set up my notebooks so that a failed run is retried when a run is resumed (in case of power failure, crashes etc.). I also implemented the final submission notebook, including the test evaluation with and without near-duplicates. In the report, I wrote the Cross-Validation, Model Comparison and Hyperparameter Tuning sections, and the near-duplicate analysis in Results.
