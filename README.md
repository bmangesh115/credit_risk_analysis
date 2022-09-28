# credit_risk_analysis

# Overview
The purpose is to analyze credit data to build models to predict credit risk. Techniques used to build models are, 
1. RandomOverSampler and SMOTE - oversampling
2. ClusterCentroids - undersampling
3. SMOTEENN - combination under and oversampling
4. BalancedRandomForestClassifer
5. EasyEnsembleClassifer

## Resources
- Data Source: LoadStats_2019Q1.csv
- Software: Jupyter Notebook 4, Python 3.7.13, Visual Studio Code 1.68.1

## Results

The link to the Python Script for the resampling.<br>
[PySpark Script for the resampling](/credit_risk_resampling.ipynb)<br>

The link to the Python Script for the ensemble.<br>
[Python Script for the ensemble](/credit_risk_ensemble.ipynb)<br>

### Credit risk predictions by different techniques

#### Naive random oversampling

1. Accuracy  = 64%

2. High risk loans
    - Precision = 0.01
    - Sensitivity = 0.62
    - F1 = 0.02

3. Low risk loans
    - Precision = 1.00
    - Sensitivity = 0.65
    - F1 = 0.79

<figure>
    <figcaption>Naive random oversampling</figcaption>
    <img src="/Resources/naive_random_oversampling.png" width=987 height=auto
         alt="Naive random oversampling">
</figure> <br>

#### SMOTE oversampling

1. Accuracy  = 63%

2. High risk loans
    - Precision = 0.01
    - Sensitivity = 0.62
    - F1 = 0.02

3. Low risk loans
    - Precision = 1.00
    - Sensitivity = 0.64
    - F1 = 0.78

<figure>
    <figcaption>SMOTE oversampling</figcaption>
    <img src="/Resources/smote_oversampling.png" width=984 height=auto
         alt="SMOTE oversampling">
</figure> <br>

#### Undersampling

1. Accuracy  = 52%

2. High risk loans
    - Precision = 0.01
    - Sensitivity = 0.60
    - F1 = 0.01

3. Low risk loans
    - Precision = 1.00
    - Sensitivity = 0.43
    - F1 = 0.60

<figure>
    <figcaption>Undersampling</figcaption>
    <img src="/Resources/unundersampling.png" width=980 height=auto
         alt="Undersampling">
</figure> <br>

#### SMOTEENN combination over and under sampling

1. Accuracy  = 65%

2. High risk loans
    - Precision = 0.01
    - Sensitivity = 0.71
    - F1 = 0.02

3. Low risk loans
    - Precision = 1.00
    - Sensitivity = 0.59
    - F1 = 0.74

<figure>
    <figcaption>SMOTEENN</figcaption>
    <img src="/Resources/unundersampling.png" width=970 height=auto
         alt="SMOTEENN">
</figure> <br>

#### BalancedRandomForestClassifier

1. Accuracy  = 79%

2. High risk loans
    - Precision = 0.04
    - Sensitivity = 0.67
    - F1 = 0.07

3. Low risk loans
    - Precision = 1.00
    - Sensitivity = 0.91
    - F1 = 0.95

<figure>
    <figcaption>BalancedRandomForestClassifier</figcaption>
    <img src="/Resources/balancedrandomforestclassifier.png" width=1011 height=auto
         alt="BalancedRandomForestClassifier">
</figure> <br>

#### EasyEnsembleAdaBoostClassifier

1. Accuracy  = 92%

2. High risk loans
    - Precision = 0.07
    - Sensitivity = 0.91
    - F1 = 0.14

3. Low risk loans
    - Precision = 1.00
    - Sensitivity = 0.94
    - F1 = 0.97

<figure>
    <figcaption>EasyEnsembleAdaBoostClassifier</figcaption>
    <img src="/Resources/easyensembleadaboostclassifier.png" width=1003 height=auto
         alt="EasyEnsembleAdaBoostClassifier">
</figure> <br>


## Summary

1. Machine learning esemble models are more accurate, precise, and sensitive compared to sampling algorithms.
2. All models has low precision for high risk loans. It flags high percentage of loans as False Positive. It will limit number of loans granted and limit customer base. Many customers with low risk profile may not get loan.
3. EasyEnsembleAdaBoostClassifier model has highest accuracy of 92%. The model has high sensitity 0.91 for high risk loans.
4. If the goal is to identify high risk loans to limit liability then EasyEnsembleAdaBoostClassifier model is recommended due to its high sensitivity.
