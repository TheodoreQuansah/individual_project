# Heart Disease Prediction Project

## Project Description

This project aims to predict the presence of heart disease in individuals based on various health-related factors. Early detection of heart disease is crucial for effective treatment and prevention. Through data exploration, statistical analysis, and machine learning modeling, we seek to uncover patterns and insights to assist healthcare professionals in identifying individuals at risk of heart disease.

## Initial Hypotheses and Questions

- Hypothesis 1: Smoking is positively correlated with the presence of heart disease.
- Hypothesis 2: Individuals with a higher BMI are more likely to have heart disease.
- Hypothesis 3: Physical activity is inversely related to heart disease risk.
- Question 1: What role does age play in heart disease?
- Question 2: Are there significant differences in heart disease prevalence among different gender and race groups?

## Data Dictionary

| Feature                | Definition | Data Type |
|:-----------------------|:-----------|:----------|
| HeartDisease           | Presence of Heart Disease (Target Variable) | Categorical (Binary) |
| BMI                    | Body Mass Index (BMI) | Numeric (Continuous) |
| Smoking                | Smoking Status (Yes/No) | Categorical (Binary) |
| AlcoholDrinking        | Alcohol Drinking Status (Yes/No) | Categorical (Binary) |
| Stroke                 | History of Stroke (Yes/No) | Categorical (Binary) |
| PhysicalHealth         | Physical Health Score | Numeric (Continuous) |
| MentalHealth           | Mental Health Score | Numeric (Continuous) |
| DiffWalking            | Difficulty Walking (Yes/No) | Categorical (Binary) |
| Sex                    | Gender (Male/Female) | Categorical |
| AgeCategory            | Age Category | Categorical |
| Race                   | Race | Categorical |
| Diabetic               | Diabetic Status (Yes/No) | Categorical (Binary) |
| PhysicalActivity       | Physical Activity Status (Yes/No) | Categorical (Binary) |
| GenHealth              | General Health Rating | Categorical |
| SleepTime              | Average Sleep Time (hours) | Numeric (Continuous) |
| Asthma                 | Asthma Diagnosis (Yes/No) | Categorical (Binary) |
| KidneyDisease          | Kidney Disease Diagnosis (Yes/No) | Categorical (Binary) |
| SkinCancer             | Skin Cancer Diagnosis (Yes/No) | Categorical (Binary) |

## Project Planning

1. **Exploratory Data Analysis (EDA)**:
   - Visualize data distributions.
   - Identify correlations and patterns.
   - Formulate initial hypotheses.

2. **Data Preprocessing**:
   - Handle missing values.
   - Encode categorical variables.
   - Split data into training, validation, and test sets.

3. **Modeling**:
   - Train machine learning models:
     - Decision Tree
     - Random Forest
     - K-Nearest Neighbors (KNN)
     - Logistic Regression

4. **Evaluation**:
   - Assess model performance using accuracy.
   - Feature importance analysis.
  
5. **Key Findings, Recommendations, and Takeaways**:

    - Smoking is strongly associated with heart disease.
    - BMI is a significant predictor of heart disease.
    - Physical activity may reduce the risk of heart disease.
    - Age plays a crucial role in heart disease risk, with older individuals at higher risk.
Further analysis is recommended to explore gender and race-based differences in heart disease prevalence.

## Instructions to Reproduce

To reproduce this project and analyze the dataset, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git


