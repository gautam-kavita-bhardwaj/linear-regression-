# Medical Insurance Cost Prediction

This project aims to predict medical insurance costs (`charges`) based on personal and demographic attributes using Machine Learning. It explores building and evaluating both standard **Linear Regression** and **Lasso Regression** models to find the best fit for the data.

---

## 📂 Project Structure

* **`insurance.csv`**: The dataset containing medical insurance records.
* **`Untitled.ipynb`**: A Jupyter Notebook containing data loading, preprocessing, feature engineering (interaction terms), and a baseline Linear Regression model.
* **`Lasso_regression.ipynb`**: A Jupyter Notebook applying Lasso Regression, including hyperparameter tuning using `LassoCV` to find the optimal alpha value.

---

## 📊 Dataset

The project uses the `insurance.csv` dataset, which includes the following 7 features:
* **age**: Age of the primary beneficiary.
* **sex**: Gender of the insurance contractor (female/male).
* **bmi**: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height.
* **children**: Number of children covered by health insurance.
* **smoker**: Smoking status (yes/no).
* **region**: The beneficiary's residential area in the US (northeast, southeast, southwest, northwest).
* **charges**: Individual medical costs billed by health insurance (Target Variable).

---

## ⚙️ Data Preprocessing & Feature Engineering

Before training the models, several preprocessing steps are applied to prepare the data:
1. **Categorical Encoding**: 
   - `sex` is mapped to binary values (`female`: 1, `male`: 0).
   - `smoker` is mapped to binary values (`yes`: 1, `no`: 0).
2. **One-Hot Encoding**: `region` is converted into dummy variables with `drop_first=True` to avoid multicollinearity.
3. **Feature Engineering**: Interaction terms are created to capture combined effects:
   - `age_smoker` = `age` * `smoker`
   - `bmi_smoker` = `bmi` * `smoker`
4. **Train-Test Split**: The data is split 80/20 into training and testing sets.

---

## 🧠 Models & Performance

Two different models are evaluated on the dataset:

1. **Multiple Linear Regression (`Untitled.ipynb`)**
   * Serves as the baseline model.
   * **Training R²:** ~0.834
   * **Test R²:** ~0.865

2. **Lasso Regression (`Lasso_regression.ipynb`)**
   * Uses `LassoCV` across a list of candidate alphas `[0.001, 0.1, 1, 2, 5, 10, 20, 30, 40, 50, 100]` with 5-fold cross-validation.
   * **Best Alpha Found:** `0.001`
   * **Test R²:** ~0.865
   * **Test MSE:** ~20,922,599

*Note: The results show that the model handles the data well without heavy overfitting, as the Training R² and Test R² are very close.*

---

## 🛠️ Prerequisites

To run these notebooks, you will need the following Python libraries installed:
* `pandas`
* `scikit-learn`
* `seaborn`
* `jupyter`

You can install them via pip:
```bash
pip install pandas scikit-learn seaborn matplotlib jupyter
