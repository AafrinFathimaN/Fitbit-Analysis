# Fitbit-Analysis

## **Project Overview**
This project analyzes Fitbit activity and sleep data to identify patterns in users' daily habits, cluster them based on activity and sleep behavior, and predict sleep quality (Good/Poor) using machine learning models. The analysis combines **clustering** and **classification** techniques to extract meaningful insights from the data.

## **Objective**
- Explore Fitbit daily activity and sleep datasets to understand user behavior.
- Group users into clusters based on activity and sleep patterns using **K-Means clustering**.
- Predict users’ sleep quality using **logistic regression** and improve performance using **SMOTE** for class balancing.
- Provide actionable insights for users or health practitioners.

## **Data**
The analysis uses two datasets from [Kaggle Fitbit Data](https://www.kaggle.com/datasets):
1. `dailyActivity_merged.csv` – contains daily activity metrics like steps, calories burned, and active minutes.
2. `sleepDay_merged.csv` – contains daily sleep metrics like total sleep duration and time in bed.

### **Features Used**
- `TotalSteps`
- `Calories`
- `VeryActiveMinutes`
- `SedentaryMinutes`
- `TotalMinutesAsleep`
- `TotalTimeInBed`
- `SleepQuality` (derived)

## **Methods**
### **Data Preprocessing**
- Merged activity and sleep datasets by user ID.
- Selected relevant features and removed duplicates.
- Handled outliers and invalid values.
- Scaled numeric features for clustering.

### **Clustering**
- Performed **K-Means clustering** to group users into 3 clusters:
  1. Sedentary Sleepers
  2. Overworked Night Owls
  3. Active Resters
- Visualized clusters using scatter plots and boxplots.
- Generated cluster summary statistics.

### **Classification**
- Derived `SleepQuality` based on sleep efficiency (Good if ≥85%).
- Built **logistic regression** models to predict sleep quality.
- Applied **SMOTE** to balance classes for better performance.
- Evaluated models using **accuracy, precision, recall, F1-score, and AUC**.
- Visualized performance using **ROC curves** and **decision trees**.

## **Results**
- Users were successfully clustered into meaningful behavior groups.
- Logistic regression achieved improved prediction accuracy after applying SMOTE.
- Key metrics and cluster summaries are exported for visualization in Power BI.

## **Files**
- `fitbit_clusters.csv` – dataset with cluster labels for each user.
- `SleepQuality_Comparison.csv` – classification metrics comparing Original vs SMOTE models.
- `Cluster_Summary.csv` – cluster-wise summary of features.
- `Fitbit_Sleep_Analysis.R` – main R script with full code and analysis.
