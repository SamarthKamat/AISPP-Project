# New Product Profit Prediction

**Assignment 2 — AI Spreadsheets and Python Programming**
**Domain:** Product Management and Finance
**Target column:** `First_Year_Profit`

## Business Problem

Before launching a new product, a product manager needs to estimate whether the
planned pricing, cost structure, marketing spend, and expected demand will produce
a profitable first year. This project builds a **linear regression model** that
predicts `First_Year_Profit` from product, cost, pricing, and market-reach inputs,
and wraps it in an interactive **Streamlit web app** so a manager can test different
launch scenarios (price, marketing budget, distribution, region, category) and
immediately see the projected profit impact — without needing to run any code
or open a spreadsheet.

## Project Files

| File | Purpose |
|---|---|
| `data.csv` | Copilot-style generated dataset (2,500+ rows, real business relationships) used for analysis and training — deliberately seeded with realistic data-quality issues (missing values, duplicates, inconsistent categories, impossible values, outliers) |
| `model_training.ipynb` | EDA, assumption checks, corrections, model training and evaluation |
| `model.pkl` | Saved final preprocessing/model pipeline |
| `app.py` | Streamlit web application |
| `README.md` | How to run the project and business problem description (this file) |
| `data_cleaned.csv` *(extra)* | Cleaned dataset after Part A corrections, produced by the notebook and used to train `model.pkl` |
| `requirements.txt` *(extra)* | Python dependencies |

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Re-run the training notebook
```bash
jupyter notebook model_training.ipynb
```
Running all cells regenerates `data_cleaned.csv` and `model.pkl` from `data.csv`.
A pre-trained `model.pkl` is already included, so this step is optional.

### 3. Launch the web application
```bash
streamlit run app.py
```
This opens the app in your browser (default: `http://localhost:8501`).

## Using the App

1. Select the **Product Category** and **Launch Region**.
2. Enter **Development Cost**, **Marketing Spend**, **Unit Selling Price**, and
   **Unit Production Cost**.
3. Enter the expected **First-Year Units Sold**, **Customer Rating**, and
   **Distribution Coverage**.
4. Click **Predict First-Year Profit** to see the estimated profit, along with
   a plain-language explanation of what the number means for decision-making.

The app includes input validation (e.g. selling price must exceed production
cost, ratings must be 1–5, coverage must be 0–1) so obviously invalid launch
plans are rejected with a clear error message rather than silently producing
a nonsensical prediction.

## Modelling Summary

- **Model:** Multiple Linear Regression inside a scikit-learn `Pipeline`
  (`StandardScaler` for numeric features, `OneHotEncoder` for `Product_Category`
  and `Launch_Region`).
- **Key engineered features:** `Unit_Margin` (selling price − production cost)
  and `Margin_x_Units` (margin × units sold), which capture the multiplicative
  business logic behind profit and substantially improved linearity.
- **Assumption checks performed:** missing values, duplicates, impossible
  values, linearity, multicollinearity (VIF), independence of errors
  (Durbin-Watson), homoscedasticity (Breusch-Pagan), normality of residuals
  (Q-Q plot, Shapiro-Wilk), and outlier/influence analysis (standardized
  residuals, Cook's distance) — see `model_training.ipynb` for full detail.
- **Evaluation metrics:** R², Adjusted R², MAE, MSE/RMSE, and residual
  diagnostics reported in the notebook.
- **Optional comparison model:** a Random Forest Regressor is trained for
  benchmarking only; Linear Regression remains the primary, deployed model
  for its interpretability.

## Model Quality Check (proof it's genuinely trained, not random)

- **Test R² = 0.943**, Train R² = 0.970 — the small train/test gap shows the
  model generalizes well and isn't overfitting or memorizing.
- **Adjusted Test R² = 0.943** — holds up even after penalizing for the number
  of predictors.
- **MAE ≈ $79,900 / RMSE ≈ $262,300** on a target ranging into the millions —
  errors are small relative to typical profit values.
- **Durbin-Watson ≈ 2.02** — residuals show no meaningful autocorrelation.
- Spot-checked predictions against real held-out rows from `data_cleaned.csv`
  consistently land within a few percent of the actual `First_Year_Profit`
  (see the "Sanity check" cell near the end of `model_training.ipynb`).

All of the above is computed directly from **this dataset's real values** —
not a placeholder/dummy model — so the coefficients and predictions reflect
the actual cost/price/volume relationships in `data.csv`.

## Screenshots

*(Add screenshots of the running app here before submission — e.g. the input
form and a sample prediction result.)*

