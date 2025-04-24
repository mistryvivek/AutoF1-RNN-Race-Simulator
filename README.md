# AutoF1 Project

This repository contains all code and datasets used for the F1 RNN simulator and analysis shown in the report. The file structure is organized by report chapter (number prefix) and order of mention within each chapter (letter suffix). Each script or notebook directly contributes to the analysis pipeline or visualization tasks as described in the report.

## 📁 Folder Structure & File Descriptions

### 📂 `3a_csv_datasets/`
This folder contains all the processed `.csv` datasets used in the analysis and in training. These datasets are collected via the `3a_create_local_dataset_from_api.py` script using the FastF1 library.

### 📄 `3a_create_local_dataset_from_api.py`
Script to collect and create local datasets by interfacing with the FastF1 library. Outputs are stored in the `3a_csv_datasets/` folder.

### 📄 `3b_data_exploration_graphs.py`
Generates initial visualizations and performs exploratory data analysis on the datasets created in 3a.

### 📄 `3c_calculate_class_weights.py`
Script for calculating class weights for categorical features such as gear, tyre compound, and pit strategy. These weights are used to handle class imbalance during model training.

### 📄 `4a_f1_dataloader.py`
A data loader script that standardizes and loads datasets for model training and testing. Handles data normalization, batching, and splitting.

### 📄 `4b_shared_functions.py`
A file containing generic methods to train and evaluate the lstm and gru.

**PLEASE RENAME FILE TO 'shared_function.py' WHEN RUNNING YOURSELF - PYTHON IMPORTS CANNOT START WITH NUMBER**

### 📄 `4c_gru.py`
A file containing the class with the implementation containing the GRU.

### 📄 `4c_lstm.py`
A file containing the class with the implementation containing the LSTM.

### 📄 `5a_autoregressive_testing.py`
Performs testing using autoregressive models to see the quality of simulations performed compared to ground truth.

### 📄 `5b_case_studies.ipynb`
Jupyter Notebook presenting detailed case studies and scenario analysis based on races at the start of the 2025 season.

## 📌 Notes

- All scripts are numbered to match their chapter in the report.
- The alphabetical suffix (a, b, c, ...) represents the order the script is referenced within that chapter.
- Make sure to run the scripts in order for best reproducibility.