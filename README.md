# Mushrooms — Edible vs Poisonous Classification + FastAPI Deployment

This repository contains a Machine Learning project based on the **Mushroom** dataset (UCI ML Repository, dataset id `73`) with the goal of predicting whether a mushroom is **edible (`e`)** or **poisonous (`p`)**.

The project includes:
- data fetching from UCI via `ucimlrepo`
- preprocessing with a custom `DataProcessor` / `DataProcessorTransformer`
- training and evaluation of multiple models (**Logistic Regression**, **XGBoost**, **Neural Network**)
- model export with `joblib`
- a **FastAPI** app exposing `/predict/*` endpoints
- dockerization


## Dataset

- Name: **Mushroom**
- Source: UCI Machine Learning Repository (fetched with `ucimlrepo`)
- Task: **binary classification**
  - `e` = edible
  - `p` = poisonous

Features are categorical attributes encoded as single-letter codes (e.g. `cap-shape`, `odor`, `habitat`, ...).  
A special value `?` can appear (notably in `stalk-root`) to represent missing/unknown values.

---

## Approach

### Preprocessing

Preprocessing is handled with a custom class:

- `DataProcessor` (data cleaning utilities)
- `DataProcessorTransformer` (a scikit-learn compatible transformer that wraps `DataProcessor` so it can be used inside `sklearn.Pipeline`)

Main steps:

1. **Missing values handling**
   - `stalk-root` is imputed with `most_frequent` (mode)

2. **Encoding**
   - binary columns are mapped to 0/1 (`bruises`, `veil-type`)
   - all other categorical columns are One-Hot Encoded (`OneHotEncoder(handle_unknown="ignore")`)

3. **Feature selection**
   - `VarianceThreshold(threshold=0.0)` to remove constant (zero-variance) features after encoding

4. **Target mapping**
   - `y = 1` if label is `p` (poisonous), else `0` (`e`)

> Note: `handle_unknown="ignore"` allows the pipeline to transform unseen categories at inference time (they will become all-zeros columns).

### Models

The repository trains and compares:

- **XGBoost** (`XGBClassifier`)
- **Logistic Regression** (`LogisticRegression`)
- **Neural Network** (Keras via **SciKeras** `KerasClassifier`)

The neural network is a small MLP (dense layers with dropout) trained with early stopping.

---

## Results

During training, models are evaluated using:

- **Accuracy**
- for NN also: **ROC-AUC**, classification report

Metrics may vary slightly depending on split and random seed.  


---

## Repository Structure

Typical structure:

- `app/`
  - `main.py` — FastAPI app with `/predict/lr`, `/predict/xgb`, `/predict/nn`
  - `model.py` — Pydantic model for request payload (with aliases like `cap-shape`)
- `models/`
  - `lr_model_pipe.joblib`
  - `xgb_model_pipe.joblib`
  - `nn_model_pipe.joblib`
  - `columns.joblib`, `col_bin.joblib`, `col_hot.joblib` (helper artifacts)
- `wrapper.py` / `data_cleaning_and_preprocessing.py` — preprocessing utilities (`DataProcessor`, `DataProcessorTransformer`)
- `nn_model.py` — contains `build_model(...)` used by SciKeras (important for serialization)
- `Dockerfile`

