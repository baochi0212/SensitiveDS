All following notebooks are run and all datasets are stored on Kaggle

"DS draft"
A draft for data exploration and initial model testing
https://www.kaggle.com/code/htnminh/ds-draft
Version 50
2253.8s - GPU P100
Input: "DS dataset" (sensitive_dev.json, sensitive_test.json,
	sensitive_train.json) (included in the repository)
Output: Logged .pkl files and .pth files (not included since
	they are for testing only)

"DS official models"
Model tuning to find the official model
https://www.kaggle.com/code/htnminh/ds-official-models
Version 24
21363.4s - GPU P100
Input: "DS dataset"
Output: "DS official models 500" (logged .pkl files and .pth
	files) (the chosen model is model_max_weighted_f1.pth,
	since the file is too large for a GitHub repository,
	this is a public link to the model
	https://www.kaggle.com/datasets/htnminh/ds-official-models-500-filtered-max-weighted-f1)

"DS Demo"
A quick demo for the official model, user input is required
https://www.kaggle.com/code/htnminh/ds-demo/edit
Input: The dataset in the link above

"DS analyze official models"
Analyze the performance of official model
https://www.kaggle.com/code/htnminh/ds-analyze-official-models
Version 3
87.2s - GPU P100
Input: "DS official models 500"
Output: All .png files in results.zip in this directory
