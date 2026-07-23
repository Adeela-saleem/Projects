# 💻 Laptop Price Predictor

A supervised machine learning project that predicts laptop prices based on hardware specifications, using **Linear Regression**. The project covers the full ML pipeline — data cleaning, EDA, encoding, model training, cross-validation — and is deployed as an interactive web app.

**🔗 Live Demo:** _[add your Streamlit app link here once deployed]_

---

## 📌 Project Overview

| | |
|---|---|
| **Algorithm** | Linear Regression |
| **Type** | Supervised Learning |
| **Problem** | Regression |
| **Dataset** | `laptopPrice.csv` (823 rows, 19 columns) |
| **Language** | Python |
| **Libraries** | Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit |
| **Author** | Adeela Saleem |

---

## ⚠️ Important Disclaimer

> This model is trained on a **historic** dataset of laptop prices. Predictions reflect the pricing patterns present in that dataset at the time it was collected — **not current, real-time market prices**. Use this tool as an estimate/reference only, not as an actual price quote.

---

## 🧹 Data Cleaning

- Removed duplicate rows
- Converted "fake object" numeric columns stored as text (e.g. `"4 GB"` → `4`, `"64-bit"` → `64`, `"3 stars"` → `3`) into proper numeric types
- Handled missing values in `processor_gnrtn` using median imputation

## 📊 Exploratory Data Analysis (EDA)

- **Univariate analysis** — distribution plots (histograms) and boxplots for all numeric features
- **Bivariate analysis** — scatter plots (numeric features vs. Price) and boxplots (categorical features vs. Price)
- **Correlation heatmaps** to identify the strongest predictors of price

**Key insight:** `ssd`, `ram_gb`, and `graphic_card_gb` showed the strongest positive correlation with Price, along with high-end processor categories (Core i7/i9, Ryzen 9) and Mac OS.

## 🔠 Encoding

- **Label Encoding** for binary columns (`Touchscreen`, `msoffice`)
- **One-Hot Encoding** (`drop_first=True`) for multi-category columns (`brand`, `processor_brand`, `processor_name`, `ram_type`, `os`, `weight`, `warranty`)

## 🎯 Target Transformation

Price was **log-transformed** (`log1p`) before training since raw price was right-skewed — this improves Linear Regression's fit, since it assumes roughly normal, symmetric errors. Predictions are transformed back with `expm1` to return real rupee values.

## 🤖 Model Training & Results

| Metric | Value |
|---|---|
| R² Score (test set) | **0.82 – 0.85** |
| Train Score | 0.88 |
| Mean Absolute Error | ~₹15,000 |
| RMSE (real ₹ scale) | **~₹20,240** |
| Average Price in dataset | ~₹76,625 |

**5-Fold Cross-Validation:**
- Mean R²: **0.85** (Std: 0.015 — consistent performance across folds, not a lucky split)

**Interpretation:** the model's typical prediction error is about **26% of the average laptop price** — a reasonable baseline for a simple Linear Regression model, with room for improvement (see below).

## 🔧 Production Pipeline

For deployment, the notebook workflow (manual scaling + separate encoding steps) was refactored into a single **scikit-learn `Pipeline`**, combining:

1. `ColumnTransformer` — One-Hot Encoding (categorical) + `StandardScaler` (numeric/binary)
2. `LinearRegression` — the model
3. `TransformedTargetRegressor` — handles the `log1p`/`expm1` transform internally

This means the saved pipeline object (`pipeline.pkl`) accepts **raw, human-readable input** and returns a **real rupee prediction** directly — no manual preprocessing needed at inference time.

## 🚀 Deployment

Built with **Streamlit** and deployed on **Streamlit Community Cloud**.

### Files in this repo

| File | Purpose |
|---|---|
| `LinearRegression.ipynb` | Full notebook — EDA, cleaning, encoding, model training, cross-validation, pipeline |
| `app.py` | Streamlit web app — takes user input and returns a price prediction |
| `pipeline.pkl` | Saved trained pipeline (preprocessing + model, generated from the notebook) |
| `requirements.txt` | Python dependencies |
| `laptopPrice.csv` | Dataset used for training |

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📈 Possible Improvements

- Remove weakly-correlated / non-generalizable features (`Number of Ratings`, `Number of Reviews` — not available for new/unreleased laptops)
- Outlier handling for extreme price values
- Try regularized models (Ridge/Lasso) to reduce overfitting
- Compare against non-linear models (Random Forest, Gradient Boosting) as a benchmark

---

*This project was built as part of a hands-on machine learning learning path, focused on understanding the complete regression workflow from raw data to a deployed, usable application.*
