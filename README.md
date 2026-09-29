# Ride Pricing Analytics 🚗

A comprehensive **Machine Learning project for ride pricing analysis** using Python and Scikit-learn to analyze historical ride data, engineer demand-supply features, develop multiple regression models, and build predictive models for historical ride-cost estimation.

The project focuses on understanding the factors influencing historical ride costs, analyzing demand-supply conditions, predicting ride costs, evaluating model performance, and explaining model predictions through model coefficients, SHAP, permutation feature importance, and individual error analysis.

---

## 🌟 Features

### Core Features

* **Data Understanding & Validation:** Inspection of data quality, data types, missing values, duplicates, categorical variables, and dataset structure.
* **Exploratory Data Analysis:** Analysis of ride-cost patterns across vehicle types, booking periods, locations, ride duration, demand-supply relationships, and numerical variables.
* **Feature Engineering:** Creation of the `Rider_Driver_Ratio` feature to represent demand-supply conditions.
* **Data Preprocessing:** Numerical scaling and categorical feature preparation using Scikit-learn pipelines.
* **Multiple Machine Learning Models:** Development and comparison of multiple regression models.
* **Hyperparameter Tuning:** Optimization of selected tree-based and boosting models using efficient randomized search.
* **Ensemble Learning:** Implementation of Voting and Stacking Regression to combine predictions from multiple models.
* **Model Evaluation:** Evaluation using MAE, RMSE, and R² along with actual-vs-predicted and residual analysis.
* **Error Analysis:** Investigation of model residuals, prediction errors, and high-error individual observations.
* **Model Explainability:** Lasso coefficients, SHAP analysis, permutation feature importance, and individual prediction analysis.
* **Experiment Tracking:** MLflow tracking of model development and evaluation experiments.
* **Business Insights:** Translation of model results into practical ride pricing and demand-supply insights.

---

## 🗂 Project Structure

The project is organized as follows:

```text
Ride-Pricing-Analytics/
│
├── data/
│   ├── raw/                            # Original dataset
│   └── processed/                      # Processed and prepared data
│
├── notebooks/
│   ├── 01_data_understanding.ipynb     # Dataset inspection and understanding
│   ├── 02_eda.ipynb                    # Exploratory Data Analysis
│   ├── 03_data_preparation.ipynb       # Data preparation and feature engineering
│   ├── 04_model_development.ipynb      # Model development and tuning
│   ├── 05_model_evaluation.ipynb       # Model evaluation and error analysis
│   ├── 06_model_comparison.ipynb       # Model comparison and benchmarking
│   └── 07_model_explainability.ipynb   # Model explainability
│
├── models/                             # Saved trained models and preprocessing artifacts
│
├── reports/
│   └── figures/                        # Project visualizations
│
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
└── .gitignore                          # Git ignored files
```

---

## 🏗️ Project Architecture

```text
+---------------------------+
|        DATA LAYER         |
|                           |
|  Raw Ride Dataset         |
|  CSV Data                 |
+-------------+-------------+
              ↓
+---------------------------+
|   DATA UNDERSTANDING      |
|                           |
|  - Dataset inspection     |
|  - Data types             |
|  - Missing values         |
|  - Duplicate analysis     |
+-------------+-------------+
              ↓
+---------------------------+
|        EDA LAYER          |
|                           |
|  - Cost distributions     |
|  - Duration analysis      |
|  - Vehicle analysis       |
|  - Booking-time analysis  |
|  - Demand-supply analysis |
|  - Correlation analysis   |
+-------------+-------------+
              ↓
+---------------------------+
|    PREPROCESSING LAYER    |
|                           |
|  - Numerical scaling      |
|  - Categorical encoding   |
|  - Train-test preparation |
+-------------+-------------+
              ↓
+---------------------------+
|   FEATURE ENGINEERING     |
|                           |
|  - Rider-driver ratio     |
|  - Demand-supply signal   |
+-------------+-------------+
              ↓
+---------------------------+
|     MODELING LAYER        |
|                           |
|  - Multiple regressors    |
|  - Hyperparameter tuning  |
|  - Voting regression      |
|  - Stacking regression    |
+-------------+-------------+
              ↓
+---------------------------+
|     EVALUATION LAYER      |
|                           |
|  - MAE                    |
|  - RMSE                   |
|  - R²                     |
|  - Residual analysis      |
|  - Error analysis         |
+-------------+-------------+
              ↓
+---------------------------+
|   EXPLAINABILITY LAYER    |
|                           |
|  - Lasso coefficients     |
|  - SHAP analysis          |
|  - Permutation importance |
|  - Local error analysis   |
+-------------+-------------+
              ↓
+---------------------------+
|     BUSINESS INSIGHTS     |
|                           |
|  Pricing patterns         |
|  Demand-supply insights   |
|  Vehicle insights         |
|  Model limitations        |
|  Pricing opportunities    |
+---------------------------+
```

---

## 🛠 Tech Stack

- **Language:** Python 3.8+
- **Environment:** Jupyter Notebook
- **Libraries:**
  - **Data Manipulation:** Pandas, NumPy
  - **Visualization:** Matplotlib, Seaborn, Plotly
  - **Machine Learning:** Scikit-learn, XGBoost, LightGBM
  - **Model Explainability:** SHAP
  - **Model Persistence:** Joblib
  - **Experiment Tracking:** MLflow
- **Version Control:** Git & GitHub

---

## 📊 Dataset Overview

The project uses a ride-sharing dataset containing **1,000 observations** and a combination of operational, customer, vehicle, booking, demand-supply, and historical pricing variables.

### Dataset Features

```text
Number_of_Riders
Number_of_Drivers
Location_Category
Customer_Loyalty_Status
Number_of_Past_Rides
Average_Ratings
Time_of_Booking
Vehicle_Type
Expected_Ride_Duration
Historical_Cost_of_Ride
```

An additional engineered feature was created during the data preparation stage:

```text
Rider_Driver_Ratio
```

### Target Variable

**Historical_Cost_of_Ride**

The objective is to predict historical ride costs using ride characteristics, demand-supply conditions, customer information, booking periods, and vehicle characteristics.

> **Note:** The target represents historical observed ride cost. It is not an explicitly defined optimal or surge-pricing target.

---

## 🎯 Problem Statement

Ride-sharing platforms operate under varying demand, supply, ride duration, vehicle, location, and booking conditions.

A ride-sharing company currently relies primarily on ride duration when determining ride fares. However, historical ride costs may also be influenced by demand-supply conditions, vehicle type, booking period, customer characteristics, and other operational factors.

The objective of this project is to develop a machine learning-based ride-cost prediction system that estimates historical ride costs using multiple ride and operational characteristics.

The project also aims to identify the most influential pricing-related features and understand situations where prediction errors remain significant.

---

## 🚀 Project Objectives

- Understand the major patterns influencing historical ride costs.
- Analyze ride costs across vehicle types, booking periods, and locations.
- Identify important relationships between expected ride duration and historical ride cost.
- Analyze demand-supply relationships between riders and drivers.
- Engineer a meaningful demand-supply feature.
- Develop and compare multiple machine learning regression models.
- Improve selected models using hyperparameter tuning.
- Evaluate Voting and Stacking ensemble approaches.
- Evaluate model performance using multiple regression metrics.
- Investigate residuals and high-error predictions.
- Explain which features contribute most to model predictions.
- Track model experiments using MLflow.
- Translate technical findings into practical ride pricing and operational insights.

---

## 🔍 Exploratory Data Analysis

The EDA stage investigates the underlying structure and behavior of the ride-sharing dataset.

### Analysis Areas

- Historical ride-cost distribution
- Expected ride-duration distribution
- Vehicle-type pricing patterns
- Expected ride duration and historical ride cost relationship
- Rider and driver relationship
- Booking-time pricing patterns
- Ride cost across duration, vehicle type, and location
- Correlation between numerical variables
- Outlier and variability analysis

The EDA stage was used to identify patterns and relationships that guided subsequent preprocessing and feature engineering decisions.

---

## ⚙️ Feature Engineering

A demand-supply feature was created to improve the model's ability to represent the relationship between riders and available drivers.

### Demand-Supply Feature

- `Rider_Driver_Ratio`

The feature is calculated using:

```text
Rider_Driver_Ratio = Number_of_Riders / Number_of_Drivers
```

This feature provides a direct representation of the relationship between rider demand and driver availability.

The feature was included because the EDA identified a meaningful relationship between the number of riders and number of drivers, while the ratio provides an additional representation of demand relative to available supply.

---

## 🤖 Machine Learning Approach

The project treats historical ride-cost prediction as a **supervised regression problem**.

The modeling workflow includes:

```text
Data Preparation
       ↓
Feature Engineering
       ↓
Train-Test Split
       ↓
Preprocessing Pipeline
       ↓
Multiple Regression Models
       ↓
Hyperparameter Tuning
       ↓
Model Comparison
       ↓
Ensemble Learning
       ↓
Final Evaluation
       ↓
Model Explainability
```

Numerical variables were standardized using **StandardScaler**, while categorical variables were transformed using **One-Hot Encoding**.

The following regression models were developed and evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Extra Trees
- Gradient Boosting
- MLP Regressor
- XGBoost
- LightGBM
- Voting Regression
- Stacking Regression

Selected tree-based and boosting models were additionally optimized using **RandomizedSearchCV**.

---

## 📈 Model Evaluation

Model performance was evaluated using three primary regression metrics:

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted historical ride costs.

### Root Mean Squared Error (RMSE)

Measures prediction error while assigning greater weight to larger errors.

### R² Score

Measures the proportion of variation in historical ride costs explained by the model.

### Evaluation Analysis

In addition to numerical metrics, the project includes:

- Actual vs Predicted Ride Cost analysis
- Residual analysis
- Prediction error analysis
- Baseline comparison
- Tuned vs untuned model comparison
- Ensemble model comparison
- Largest prediction error investigation

### Final Model Metrics

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Lasso Regression | 52.6056 | 67.5878 | 0.8747 |
| Linear Regression | 52.6138 | 67.5885 | 0.8747 |
| Ridge Regression | 52.5943 | 67.5893 | 0.8747 |
| Stacking Regression | 52.0649 | 67.7824 | 0.8740 |
| Voting Regression | 52.9107 | 69.6303 | 0.8670 |
| MLP Regressor | 55.5282 | 69.8909 | 0.8660 |
| Gradient Boosting Tuned | 53.3413 | 69.9215 | 0.8659 |
| XGBoost | 54.2290 | 71.9309 | 0.8581 |
| Extra Trees | 53.0204 | 72.0363 | 0.8577 |
| Gradient Boosting | 53.6769 | 72.1217 | 0.8573 |
| LightGBM Tuned | 54.7664 | 72.1934 | 0.8571 |
| XGBoost Tuned | 54.7342 | 73.5117 | 0.8518 |
| Random Forest | 55.6138 | 74.6436 | 0.8472 |
| Random Forest Tuned | 63.3565 | 78.9125 | 0.8292 |
| LightGBM | 57.9706 | 79.1346 | 0.8282 |
| Extra Trees Tuned | 66.9269 | 83.8946 | 0.8070 |

The mean-based baseline produced an MAE of approximately **160.96**, RMSE of approximately **191.15**, and R² of approximately **-0.002**.

All trained models substantially outperformed the baseline on the held-out test dataset.

---

## 🔎 Model Explainability

Model explainability was performed using **Lasso coefficients**, **SHAP**, **permutation feature importance**, and individual prediction analysis.

### Top Important Features

The strongest features identified through the explainability analysis were:

| Rank | Feature |
| ---: | --- |
| 1 | Expected Ride Duration |
| 2 | Vehicle Type - Premium |
| 3 | Vehicle Type - Economy |
| 4 | Rider-Driver Ratio |
| 5 | Number of Drivers |
| 6 | Average Ratings |
| 7 | Number of Riders |
| 8 | Number of Past Rides |
| 9 | Location Category |
| 10 | Customer Loyalty Status |

The analysis consistently identifies **Expected Ride Duration** as the dominant predictive signal across the evaluated explainability techniques.

Vehicle type provides the next notable contribution, while demand-supply and other operational variables provide additional predictive information.

### SHAP Analysis

SHAP analysis was performed on the tuned Gradient Boosting model to understand how individual features contribute to ride-cost predictions.

Expected Ride Duration had a mean absolute SHAP value of approximately **150.37**, substantially higher than the remaining features.

### Permutation Feature Importance

Permutation importance also identified Expected Ride Duration as the dominant feature, with an importance value of approximately **159.33**.

The consistency between SHAP and permutation importance provides additional support for the importance of ride duration within the current dataset.

---

## 💡 Key Findings

- Expected Ride Duration has a very strong positive relationship with historical ride cost, with a correlation of approximately **0.93**.
- Premium rides generally have higher historical costs than Economy rides.
- Number of Riders and Number of Drivers have a moderate positive correlation of approximately **0.63**.
- Booking periods show substantial overlap in historical ride-cost distributions.
- Rider_Driver_Ratio provides an additional representation of demand-supply conditions.
- Expected Ride Duration is consistently identified as the strongest predictive feature across the explainability analysis.
- Vehicle type provides additional predictive information beyond ride duration.
- All trained models substantially outperform the mean-based baseline.
- Lasso Regression records the lowest RMSE and highest R² among the evaluated models.
- Stacking Regression records the lowest MAE among the evaluated models.
- Hyperparameter tuning produces mixed results, with some tuned models performing better and others performing worse than their untuned versions.
- Individual prediction analysis shows that substantial errors can still occur despite strong overall test-set performance.

---

## ⚠️ Limitations & Future Improvements

### Current Limitations

The dataset contains only **1,000 observations**, which limits the amount of information available for learning more complex pricing patterns.

The target variable represents **historical observed ride cost** rather than an explicitly defined optimal dynamic price.

The dataset does not contain:

- Real-time surge pricing
- Actual optimal fares
- Traffic conditions
- Weather conditions
- Geographic demand concentration
- Events or local demand indicators
- Detailed time-series pricing information

Expected Ride Duration is also the dominant predictive feature, indicating that the current dataset may not contain enough additional variables to fully represent real-world dynamic pricing behavior.

Individual prediction errors can remain substantial despite strong aggregate model performance.

### Future Improvements

- Introduce larger and more diverse ride-sharing datasets.
- Incorporate real-time rider demand and driver availability.
- Add detailed geographic and location-based features.
- Incorporate traffic and weather conditions.
- Include holidays, events, and local demand indicators.
- Add historical surge-pricing information.
- Develop a dedicated optimal-price or surge-price target.
- Explore real-time pricing and demand forecasting techniques.
- Evaluate model performance using time-aware validation when chronological data becomes available.
- Develop model monitoring and prediction-drift analysis for production deployment.

---

## 📁 Notebook Workflow

The project is divided into seven notebooks to maintain a structured and reproducible workflow.

| Notebook | Purpose |
| --- | --- |
| `01_data_understanding.ipynb` | Dataset inspection and initial understanding |
| `02_eda.ipynb` | Exploratory Data Analysis and visualization |
| `03_data_preparation.ipynb` | Data preparation, preprocessing, and feature engineering |
| `04_model_development.ipynb` | Model development, tuning, and ensemble learning |
| `05_model_evaluation.ipynb` | Model performance and error analysis |
| `06_model_comparison.ipynb` | Model benchmarking and comparative analysis |
| `07_model_explainability.ipynb` | Feature importance, SHAP, and prediction analysis |

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd Ride-Pricing-Analytics
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Virtual Environment

#### Windows

```bash
.\.venv\Scripts\activate
```

#### Mac/Linux

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the `notebooks/` directory and run the notebooks in numerical order.

---

## 📌 Project Status

**Status:** Completed ✅

The project currently includes:

- Data understanding
- Exploratory data analysis
- Outlier analysis
- Data preprocessing
- Feature engineering
- Multiple regression models
- Hyperparameter tuning
- Voting ensemble
- Stacking ensemble
- MLflow experiment tracking
- Model evaluation
- Residual analysis
- Model comparison
- SHAP explainability
- Permutation feature importance
- Individual prediction analysis
- Business insights

---

## 👨‍💻 Author

**Tejas Jadhav**

- GitHub: [@tejas-jadhav](https://github.com/TejasMJ)
- LinkedIn: [Tejas Jadhav](https://www.linkedin.com/in/Tejas-M-Jadhav/)
