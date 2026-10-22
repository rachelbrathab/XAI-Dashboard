# XAI Model Explainability Dashboard

An interactive Streamlit application that lets users **upload their own CSV dataset**, train a Random Forest classifier, evaluate its performance, and explore why it makes predictions using Explainable AI (XAI).

> **Educational use only:** explanations describe the model's behaviour; they do not prove causation or guarantee that a prediction is correct.

## Start here

The Home page is intentionally focused on uploading a dataset. Upload a CSV, choose the target column, and select **Analyze dataset**. Once training completes, the remaining dashboard sections become available.

### Dataset requirements

- CSV file with a header row and at least 10 rows.
- Select one target column containing at least two classes, with at least two rows per class.
- Include at least one numeric predictor column.
- The target may be text or numeric. It is encoded for classification and shown using its original class labels in the interface.
- Numeric predictor columns are used. Non-numeric predictor columns are excluded and listed in the Dataset Explorer; they are not automatically encoded in this version.
- Rows with missing/infinite values in the selected target or numeric predictors are removed before splitting.
- Upload size is limited to 50 MB, with a maximum of 100,000 rows and 200 columns for responsive interactive use.

The application does not load a built-in dataset or depend on a specific medical dataset. Use data that you are authorized to analyze, and avoid uploading personal or sensitive information unless you have appropriate permission.

## Dashboard sections

- **Home / Upload:** upload a CSV, preview it, select the target, and train the model.
- **Dataset Explorer:** inspect usable rows, feature statistics, target distribution, and excluded non-numeric columns.
- **Model Performance:** view accuracy, macro precision/recall/F1, confusion matrix, and classification report on a held-out test set.
- **Global Explainability:** inspect global SHAP summaries and permutation feature importance.
- **Local Explainability:** select a test observation and inspect class-aware SHAP waterfall and LIME explanations.
- **Feature Behaviour:** compare Partial Dependence (PDP) and Individual Conditional Expectation (ICE) plots.
- **XAI Comparison:** compare the questions each technique can answer.

## XAI methods

### SHAP

SHAP values attribute a model output to input features. Global views summarize feature contributions over the test set; local waterfall plots explain the selected observation and its predicted class.

### LIME

LIME perturbs an individual observation and fits a locally weighted surrogate to approximate the classifier's behaviour around that observation. It is a local explanation and can vary with its sampling settings.

### Permutation feature importance

This method shuffles one feature in the held-out test data and measures the resulting change in accuracy. Larger decreases suggest greater reliance by the fitted model; correlated features can affect interpretation.

### PDP and ICE

Partial Dependence plots summarize the average model response as one feature varies. ICE plots show how the response varies across individual observations. They describe model behaviour, not causal effects; correlated features can produce unrealistic combinations when varied independently.

ALE was considered as an optional technique but is not implemented. The five methods above provide complementary global, local, and feature-response views without adding another dependency.

## Model and evaluation methodology

The app uses a `RandomForestClassifier` with 200 trees and `random_state=42`. Data is split into 80% training and 20% test subsets using a stratified split and `random_state=42`. The model is trained only on the training subset. Accuracy, macro precision, macro recall, macro F1-score, a confusion matrix, and a per-class report are calculated from the held-out test data.

This is a general tabular classification demonstration, not a clinical, financial, or other high-stakes decision system. Performance depends on the uploaded dataset, its quality, class balance, feature selection, and train/test split.

## Installation

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run the app

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, upload a CSV on the Home page, choose the target column, and click **Analyze dataset**. The model is trained in memory for the current analysis; the app does not load or overwrite a pre-trained model artifact.

## Tests and checks

```bash
python -m pytest -q
python -m compileall -q app.py src utils tests
```

The test suite covers the original dataset-loader utilities as well as uploaded-data validation, model training, evaluation, SHAP, LIME, permutation importance, PDP, and ICE.

## Project structure

```text
.
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── explainability.py
│   └── model.py
├── tests/
│   ├── conftest.py
│   ├── test_data_loader.py
│   ├── test_evaluation.py
│   ├── test_explainability.py
│   └── test_model.py
└── utils/
    └── visualization.py
```

## Limitations

- This version is designed for tabular **classification** datasets with numeric predictors. Non-numeric predictor columns are excluded rather than encoded.
- Missing and infinite values in selected columns are removed row-wise; a dataset with many such values may lose a substantial number of rows.
- Very small datasets or target classes with too few examples may not support a reliable stratified split.
- Model explanations describe fitted-model behaviour, not causation or guaranteed correctness.
- Correlated features can share importance and complicate PDP/ICE interpretation.
- The current interface is optimized for local interactive analysis and is not a multi-user model hosting service.

## References

- [scikit-learn classification and model evaluation](https://scikit-learn.org/stable/supervised_learning.html)
- [SHAP: A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- [LIME: Why Should I Trust You?](https://arxiv.org/abs/1602.04938)
- [scikit-learn partial dependence documentation](https://scikit-learn.org/stable/modules/partial_dependence.html)
