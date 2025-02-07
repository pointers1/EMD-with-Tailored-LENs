# Explainable-Malware-Detection-With-Tailored-Logic-Explained-Networks
Introduction

This repository contains code for conducting various experiments as described in the paper "Explainable Malware Detection with Tailored
Logic Explained Networks". The following scripts are provided for different types of experiments:

- **`LEN.py`**: The primary script for experiments comparing with black-box models using feature sizes of 10, 50, 100, 200, 500, 1000, and 2000.
- **`compare_with_cl.py`**: Script for comparison with Concept Learning approaches using feature sizes of 25, 50, 75, 100, and 200.
- **`compare_with_dtree.py`**: Used for comparison with decision tree algorithms.
- **`explanations_exp.py`**: Used for experiments on evaluating and comparing explanations with feature sizes of 5, 10, 15, 20, and 25.

## Features and File Descriptions

- **`features.txt`**: A text file containing dictionaries of selected features for different feature sizes. The features were selected based on the Decision Tree algorithm and the ESFS algorithm as described in the referenced paper.
- **`malware_dataset.csv`**: The dataset used for the experiments. Ensure this file is in the same directory as the script files.

## Experiment Details
Ensure you have the necessary dependencies installed (check requirements file).

(1). Comparison with Black Box Models

#### Feature Sizes
The following feature sizes were used: 10, 50, 100, 200, 500, 1000, and 2000.

#### Running the Experiment
1. Open `LEN.py`.
2. Modify the `num_features` variable to the desired feature size.
3. Execute the script: python LEN.py
4. Repeat for other feature sizes by changing the `num_features` value and re-running the script.

(2). Comparison with Concept Learning Approaches

#### Running the Experiment
1. Open `compare_with_cl.py`.
2. Execute the script**: python compare_with_cl.py

(3). Comparison with Decision Tree Algorithm

#### Feature Sizes
The following feature sizes were used: 25, 50, 75, 100, and 200.

#### Running the Experiment
1. Open `compare_with_dtree.py`.
2. Modify the `num_features` variable** to the desired feature size (25, 50, 75, 100, or 200).
3. Execute the script: python compare_with_dtree.py
4. Repeat for other feature sizes by changing the `num_features` value and re-running the script.

(4). Evaluating and Comparing Explanations

#### Feature Sizes
The following feature sizes were used: 5, 10, 15, 20, and 25.
### To Run the experiments:
   1. Open `explanations_exp.py`.
   2. Modify the `num_features` variable to the desired feature size (5, 10, 15, 20, or 25).
   3. Execute the script: python explanations_experiment.py
   4 Repeat the above steps for each feature size you want to evaluate.


## Notes

- The various features selected based on the Decision Tree algorithms have been saved in the `features.txt` file as dictionaries. Each script will automatically read the appropriate features from this file based on the `num_features` value you set.
- For comparison with the decision tree algorithm, the dataset used in the paper (Interpretable Rules with a Simplified Data Representation - a Case Study with the EMBER dataset) with features extracted using the ESFS algorithm are in the folder "dtree_data"
- Ensure that the dataset and the `features.txt` file are in the same directory as the scripts or adjust the paths accordingly.

## Conclusion

By following these instructions, you can effectively run and evaluate experiments on different feature sizes using the provided scripts. 

For any issues or questions, please contact peter.anthony@fmph.uniba.sk.
