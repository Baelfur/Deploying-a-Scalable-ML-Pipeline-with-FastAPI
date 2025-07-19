# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
 - Model Type: Random Forest Classifier
 - Framework: scikit-learn v1.x
 - Trained Usage: Tabular US census income dataset (adult income prediction)
 - Version: v1.0
 - Input Features
    - Categorical: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native_country`
    - Continuous: `age`,`education-num`,`capital-gain`,`capital-loss`,`hours-per-week`
 - Output: Binary label - whether income > 50k

## Intended Use

- Designed for educational use in MLOps pipeline development
- Intended to demonstrate CI/CD, slice-based model validation, and API deployment.
- Not recommended for production use or real-world income prediction

## Training Data

 - Data Source: UCI Adult Census dataset
 - Number of rows: ~32,000 (after cleaning)
 - Label distribution:
    - >50K: ~25%
    - <=50K: ~75%

- Preprocessing:
    - OneHotEncoding for categorical features
    - LabelBinarizer for target
    - No imputation performed (rows with missing values were dropped)

## Evaluation Data
 - 20% test split (stratified)
 - Same preprocessing pipeline as training data
 - Evaluation includes:
    - Overall model metrics
    - per-slice performance based on categorical feature groups

## Metrics
| Metric    | Value  |
| --------- | ------ |
| Precision | 0.7419 |
| Recall    | 0.6384 |
| F1 Score  | 0.6863 |

## Ethical Considerations
 - Bias Risk: Model may inherit biases present in census data
 - Socioeconomic Inference: Predicting income categories may reinforce stereotypes.
 - Data Representation: Dataset underrepresents some groups, may impact fairness

## Caveats and Recommendations
 - Model is not validated for production or real-world deployment.
 - Slice analysis should be reviewed to ensure fair treatment across subgroups.
 - Users should not deploy this model to make actual hiring, lending, or policy decisions
 - Consider using differential privacy or fairness-aware training for real-world deployment scenarios.