# CSNLP-ETH
Computational Semantics project

* The full code used to train the OT-based models in available in the `model_code` directory.

* The pre-trained OT-based models can be downloaded here: https://polybox.ethz.ch/index.php/s/krYnXS3FJWNS2aM.
There are 35 models in total (average size 737MB). The model filenames correspond to the conventions used in the paper.

* All evaluation scripts (for OT-based models) are provided in the `evaluation_scripts` directory.

* The primary dataset is available in the `pan25_data` directory. The StackExchange dataset used for further evaluation is available in the `pan22_data` directory. Note that only `dataset3/validation` was used in the study.

* The `requirements.txt` file is taken verbatim from the `venv` used for model training on the student cluster. There is also a sample batch file (`sample_sbatch_run.sbatch`) that served as a template for all training runs.

* All plots are available in `plots`. The full code used to create those plots is available in `plot_code`.

* The files `count_labels_train.py` and `count_labels_validation.py` have been used to count the number of total labels in the `train` and `validation` sets from `pan25_data`, respectively.

* Code of the factorized attention model & its tuning / ablation study: https://colab.research.google.com/drive/10HSKr8ka7Nivxhipu6fg1TovVV8j4hOc?usp=sharing

* Link to the factorized attention model: https://drive.google.com/file/d/1yMx-brh16xzph3grfJtQ6hI4MjQG_XM1/view?usp=drive_link
