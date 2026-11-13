# Customer Churn Intelligence

An end-to-end machine learning project that predicts which telecom customers will leave, **explains why**, scores whole customer lists, and estimates the **money** a retention campaign would earn.

Dataset: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 21 columns).

## Features

| Tab | What it does |
|---|---|
| 🔮 **Prediction** | Live churn probability for one customer, risk gauge, **"Why this score?"** explanation (sampled Shapley values), rule-based retention tips, and what-if charts (contract type, tenure) |
| 📦 **Batch Scoring** | Upload a CSV/Excel list, validate it, score every customer, rank a **priority call list** by revenue at risk, and download CSV/Excel |
| 💰 **Business Impact** | Profit curve and strategy comparison using editable assumptions (offer cost, success rate, revenue horizon). Finds the most profitable threshold |
| 📊 **Data Dashboard** | Filterable churn analytics (contract, internet service, payment method, tenure, charges) |
| 🧠 **Model Insights** | Test metrics and global feature importance |

## Machine learning workflow (`main.ipynb`)

1. Load and inspect, then clean (`TotalCharges` text to number, duplicates removed)
2. EDA
3. Stratified 80/20 train/test split **before** any fitting
4. `ColumnTransformer` (scale numbers, one-hot encode text) inside a scikit-learn `Pipeline`
5. Logistic Regression, Decision Tree and Random Forest tuned with `GridSearchCV` (5-fold, ROC-AUC, class-balanced)
6. Model chosen by **cross-validation**, test set used once
7. Save one pipeline file, metrics, and the scored test set

Because scaling and encoding live inside the saved pipeline, the app feeds it raw customer values and training/serving can never drift apart.

## Project structure

```
churn-app/
├── main.ipynb                 # training notebook (creates the files below)
├── app.py                     # Streamlit dashboard
├── churn_utils.py             # tested helper functions (scoring, explanations, profit curve)
├── tests/test_churn_utils.py  # unit tests
├── requirements.txt
├── Telco-Customer-Churn.csv   # dataset
├── churn_pipeline.pkl         # created by the notebook
├── model_metrics.json         # created by the notebook
└── test_scored.csv            # created by the notebook
```

## Run it locally

```bash
# 1. In JupyterLab: open main.ipynb -> Kernel -> Restart Kernel and Run All Cells
# 2. In Anaconda Prompt, from the project folder:
streamlit run app.py
```

Use the same Python environment for the notebook and the app, so the saved model loads under the scikit-learn version that created it.

## Run the tests

```bash
pip install pytest
python -m pytest -q
```

## Deploy for free (Streamlit Community Cloud)

1. Put this folder on GitHub (keep the `.pkl`, `.json` and `.csv` files, they are small).
2. In `requirements.txt`, set `scikit-learn==` to the version printed by the notebook's first cell.
3. Go to https://share.streamlit.io, sign in with GitHub, choose **New app**, pick the repo, set the main file to `app.py`, and deploy.

## How the key features work

- **Explanations:** for many random feature orders, start from a typical customer and switch features to this customer's values one group at a time. The change in predicted probability is credited to the feature. Averaging gives each feature's contribution, and *typical-customer probability + all contributions = this customer's probability*. No extra library needed.
- **Business impact:** for every customer at or above the threshold, cost = offer cost, and benefit (only if the customer really would have churned) = success rate x monthly charges x months saved. Net benefit = benefit - cost, computed on held-out test customers.

## Limitations

- One public dataset and one train/test split. Results are illustrative, not a guarantee for a real company.
- Importance and explanations describe what the model uses, **not what causes churn**.
- The most profitable threshold is chosen on the same test data, so its benefit is slightly optimistic.
- Business assumptions in the sidebar are placeholders. Replace them with real figures.
- Class-balanced training makes predicted probabilities higher than true frequencies, so treat them as risk scores rather than exact probabilities.

## Screenshots

_Add screenshots of each tab here after you run the app._
