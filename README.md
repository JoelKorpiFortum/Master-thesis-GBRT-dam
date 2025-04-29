# Explainable Gradient Boosted Tree Models for Hydropower Dam Monitoring
Master Thesis Project VT25 by Joel Korpi

## Data
Due to confidentiality reasons, the dam data is not included in this repo. However, all the code for preprocessing exists within `utils/data_preparation.py`

## Setting up a virtual environment
To create a Conda environment using the `environment.yml` file in this repository, follow these steps:

1. Open a terminal or command prompt.

2. Navigate to the directory where the `environment.yml` file is located.

3. Run the following command to create the Conda environment:

```conda env create -f environment.yml```

4. Conda will now download and install all the necessary packages specified in the environment.yml file.

5. Once the environment creation process is complete, activate the environment using the following command:

```conda activate gbrt_thesis```

You have successfully created and activated the Conda environment using the `environment.yml` file.

## Hyperparameter tuning

For the project he optimization of the models was performed in AWS SageMaker, where the execution is done via the notebook `SageMakerExecution.ipynb`.
If the models should be tuned, there is a bool variable called `tuning` inside the main of each respective file of the boosted tree models 
(`GBRT.py`, `XGBoost.py`, `LightGBM.py`) that needs to be set to **True** if the goal is to tune the hyperparameters. If set to false,
only inference will be performed and the saved hyperparameters are fetched from `models/best_params`.


