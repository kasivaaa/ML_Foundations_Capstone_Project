# ML_Foundations_Capstone_Project
# Student Academic Performance Prediction
## Problem Statement

Student academic performance is influenced by a combination of academic, social, and personal factors. However, educational institutions often rely on students’ past grades or general observations to identify students who may be struggling academically. This can make it difficult to identify students at risk of poor performance early enough to provide appropriate academic support.

This project aims to use machine learning techniques to analyze student-related factors and predict academic performance. The study will investigate how factors such as study time, previous academic performance, attendance, failures, parental background, and other relevant student characteristics are associated with students’ final academic outcomes.

The project will develop and compare machine learning models to predict students’ final academic scores and, where appropriate, classify students into different performance categories. The models will be evaluated using appropriate performance metrics, followed by error analysis to understand where and why the models make incorrect predictions.

The ultimate goal is to determine whether machine learning can provide useful insights into student performance and help identify factors that may be associated with academic success or difficulty. These findings could support educators in making more informed decisions about academic intervention and student support.
## 1. Research Questions
Main Research Question

How effectively can machine learning models predict students’ academic performance using their academic, demographic, and behavioral characteristics?

Specific Research Questions
Which academic, demographic, and behavioral factors are most strongly associated with students’ final academic performance?
How do factors such as study time, previous grades, failures, and attendance relate to students’ final scores?
How accurately can machine learning models predict a student’s final academic score?
Which machine learning model provides the best predictive performance for student academic outcomes?
What types of prediction errors does the best-performing model make, and are there identifiable patterns in those errors?
Can the model help identify students who may be at risk of poor academic performance?

These questions are good because they don't just ask "which model has the highest score?" They also give you something meaningful to investigate during EDA, modelling, and error analysis.

## 2. Project Objectives
General Objective

To develop and evaluate machine learning models for predicting student academic performance using relevant academic, demographic, and behavioral characteristics.

Specific Objectives
To understand and clean the student academic performance dataset by identifying missing values, inconsistent data, duplicates, and other data-quality issues.
To perform exploratory data analysis (EDA) to identify patterns, trends, relationships, and potential factors associated with student academic performance.
To identify and appropriately handle outliers in the dataset before model development.
To prepare the dataset for machine learning by applying appropriate categorical encoding and numerical feature scaling.
To develop a preprocessing pipeline that combines the required data transformations and helps prevent data leakage.
To split the dataset into training and testing sets and train multiple machine learning models.
To compare the performance of different machine learning models using appropriate evaluation metrics.
To conduct error analysis on the selected model to identify patterns in incorrect or inaccurate predictions.
To identify the most influential factors associated with student academic performance based on the developed models.
To provide data-driven recommendations that could help educators identify students who may require additional academic support.
## Project Overview

This project focuses on the application of machine learning techniques to predict and understand student academic performance. Student performance can be influenced by several factors, including previous academic results, study habits, attendance, failures, family background, and other demographic and behavioral characteristics. Understanding these factors can help educational institutions identify students who may require additional academic support.

The project will use a student academic performance dataset containing information about students and their academic and personal characteristics. The dataset will first be examined and cleaned to ensure that it is suitable for analysis and machine learning. Exploratory Data Analysis (EDA) will then be conducted to understand the distribution of the data, identify relationships between variables, investigate potential outliers, and determine factors that may be associated with academic performance.

After the exploratory analysis, the data will be prepared for machine learning. Categorical variables will be encoded, numerical variables will be appropriately scaled, and these transformations will be combined into a preprocessing pipeline. The dataset will then be divided into training and testing sets to ensure that the models are evaluated on data they have not previously seen.

Multiple machine learning models will be trained and evaluated to determine which approach provides the most reliable predictions of student academic performance. Appropriate evaluation metrics will be used to compare the models, and the best-performing model will be subjected to further error analysis. Residual and prediction-error analysis will be used to investigate whether the model's errors show any meaningful patterns.

Finally, the project will interpret the model results and identify the factors that are most relevant to student academic performance. The findings will be used to provide data-driven recommendations that could assist educators and educational institutions in identifying students who may be at risk of poor performance and in developing appropriate academic interventions.

The project therefore combines data analysis and machine learning to move beyond simply describing student performance and instead explore whether student outcomes can be predicted and what factors may contribute to those predictions.


## Data Structure

The dataset contains information about students' demographic characteristics, family background, academic history, school-related activities, and final academic performance.

| Variable | Description | Data Type | Role |
|---|---|---|---|
| `school` | Student's school | Categorical | Feature |
| `sex` | Student's gender | Categorical | Feature |
| `age` | Student's age | Numerical | Feature |
| `address` | Type of residential area | Categorical | Feature |
| `famsize` | Family size | Categorical | Feature |
| `Pstatus` | Parents' cohabitation status | Categorical | Feature |
| `Medu` | Mother's education level | Ordinal | Feature |
| `Fedu` | Father's education level | Ordinal | Feature |
| `Mjob` | Mother's occupation | Categorical | Feature |
| `Fjob` | Father's occupation | Categorical | Feature |
| `reason` | Reason for choosing the school | Categorical | Feature |
| `guardian` | Student's guardian | Categorical | Feature |
| `traveltime` | Travel time to school | Ordinal | Feature |
| `studytime` | Weekly study time | Ordinal | Feature |
| `failures` | Number of previous class failures | Numerical | Feature |
| `schoolsup` | Extra educational support | Categorical | Feature |
| `famsup` | Family educational support | Categorical | Feature |
| `paid` | Extra paid classes | Categorical | Feature |
| `activities` | Participation in extracurricular activities | Categorical | Feature |
| `nursery` | Attended nursery school | Categorical | Feature |
| `higher` | Desire to pursue higher education | Categorical | Feature |
| `internet` | Internet access at home | Categorical | Feature |
| `romantic` | In a romantic relationship | Categorical | Feature |
| `famrel` | Quality of family relationships | Ordinal | Feature |
| `freetime` | Amount of free time | Ordinal | Feature |
| `goout` | Frequency of going out with friends | Ordinal | Feature |
| `Dalc` | Workday alcohol consumption | Ordinal | Feature |
| `Walc` | Weekend alcohol consumption | Ordinal | Feature |
| `health` | Current health status | Ordinal | Feature |
| `absences` | Number of school absences | Numerical | Feature |
| `G1` | First-period grade | Numerical | Feature* |
| `G2` | Second-period grade | Numerical | Feature* |
| `G3` | Final grade | Numerical | Target |
