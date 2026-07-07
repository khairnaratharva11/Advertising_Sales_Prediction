# Advertising Sales Prediction

## 📌 Overview
This repository contains a machine learning project that predicts product sales based on advertising budgets allocated to TV, Radio, and Newspapers. The project uses a **Linear Regression** model and is structured to demonstrate an incremental approach to machine learning—starting from basic data cleaning and moving through Exploratory Data Analysis (EDA), model building, evaluation, and visualization.

## 📊 The Dataset
The project utilizes the `Advertising.csv` dataset, which contains the following features:
* **TV:** Advertising budget for TV (in thousands of dollars)
* **radio:** Advertising budget for Radio (in thousands of dollars)
* **newspaper:** Advertising budget for Newspaper (in thousands of dollars)
* **sales:** Product sales (in thousands of units) - *Target Variable*

## 🚀 Incremental Approach
The code is split into multiple scripts to showcase the step-by-step evolution of the model:

* **`Advertising1.py` (Basic Data Loading):** Focuses on the initial steps of loading the dataset using Pandas, inspecting the data shape, and performing basic data cleaning (removing the unnecessary `Unnamed: 0` index column).
* **`AdvertisingCaseStudyAnalysis.py` (Exploratory Data Analysis):** Expands on the data loading phase by adding comprehensive EDA. It checks for missing values, generates statistical summaries, and calculates the correlation matrix to understand the relationships between advertising channels and sales.
* **`Advertising2.py` (Baseline Model):** Introduces the `scikit-learn` library to split the data into training and testing sets (90/10 split) and trains the initial Linear Regression model, outputting the basic coefficients and intercept.
* **`AdvertisingCaseStudyModelBuilding.py` (Model Evaluation):** Builds upon the baseline model by introducing formal evaluation metrics. It calculates the Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared ($R^2$) values to quantify the model's accuracy.
* **`AdvertisingCaseStudyModelBuildingVisualization.py` (Final Pipeline & Visualization):** The final iteration of the code. It combines all previous steps and utilizes `matplotlib` to plot the Actual Sales versus the Predicted Sales, providing a clear visual representation of the model's performance.

## 🛠️ Technologies Used
* **Language:** Python 3
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`LinearRegression`, `train_test_split`, metrics)
* **Visualization:** Matplotlib

## ⚙️ How to Run
1. Clone the repository to your local machine.
2. Ensure you have the required libraries installed: `pip install pandas numpy scikit-learn matplotlib`.
3. Make sure the `Advertising.csv` dataset is in the same directory as the scripts.
4. Run the scripts sequentially to see the pipeline evolve, or execute the final visualization script directly:
   ```bash
   python AdvertisingCaseStudyModelBuildingVisualization.py
