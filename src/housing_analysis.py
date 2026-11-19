"""Hong Kong Housing Data Analysis

Cleaned course analysis. Run as a script or cell-by-cell in a percent-script IDE.
"""

# %% Load public data
import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/just4jc/HKProp/master/HKProp_Dataset.csv"
df = pd.read_csv(url)
print(df.head())
print(df.describe())
df.info()
print(df.isnull().sum())

# %% Data cleaning
df = df.drop(columns=["Tower", "Rehab_Year", "Flat"])
df["Bed_Room"] = df["Bed_Room"].fillna(df["Bed_Room"].median())
df = df.dropna(subset=["SaleableArea"])

cols_cleaning = ["Gross Area_Price", "SaleableAreaPrice", "Gross Area", "SaleableArea"]
for col in cols_cleaning:
    df[col] = (df[col].astype(str).str.replace("$", "", regex=False)
               .str.replace(",", "", regex=False).replace("--", "0").astype(float))

df["Roof"] = df["Roof"].map({"N": 0, "Y": 1})
df["Prop_Type"] = df["Prop_Type"].str.strip().str.lower()
df = pd.get_dummies(df, columns=["Prop_Type"], drop_first=True)
df["Prop_Type_single"] = df["Prop_Type_single"].astype(int)
df["Reg_Date"] = pd.to_datetime(df["Reg_Date"])
df["Reg_Year"] = df["Reg_Date"].dt.year
df["Reg_Month"] = df["Reg_Date"].dt.month
df = df.drop(columns=["Reg_Date"])

Q1 = df["SalePrice_10k"].quantile(0.25)
Q3 = df["SalePrice_10k"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_clean = df[df["SalePrice_10k"].between(lower_bound, upper_bound)].copy()
print("Dataset shape after outlier removal:", df_clean.shape)

# %% Exploratory analysis
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
sns.heatmap(df_clean.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()

corr_with_price = df_clean.corr(numeric_only=True)["SalePrice_10k"].drop("SalePrice_10k")
top_5 = corr_with_price.reindex(corr_with_price.abs().sort_values(ascending=False).index).head(5)
print("Top 5 correlations")
for feature, corr_val in top_5.items():
    print(f"{feature:<20}: {corr_val:+.3f}")

key_vars = ["SaleableArea", "Floor", "Build_Ages", "Longitude"]
plt.figure(figsize=(15, 8))
for i, var in enumerate(key_vars, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df_clean[var], kde=True)
    plt.title(f"Distribution of {var}")
plt.tight_layout()
plt.show()

# %% Feature engineering and multicollinearity
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

df_clean = df_clean.drop(columns=["Prop_Name_ENG", "ADDRESS_ENG"])
target = "SalePrice_10k"
X = df_clean.drop(columns=[target])
y = df_clean[target]

def print_vif(frame):
    result = pd.DataFrame({
        "Feature": frame.columns,
        "VIF": [variance_inflation_factor(frame.values, i) for i in range(frame.shape[1])]
    })
    print(result.sort_values("VIF", ascending=False).to_string(index=False, float_format="%.2f"))

print_vif(X)
X = X.drop(columns=["Gross Area_Price", "SaleableAreaPrice", "Gross Area", "Estate_Size", "Reg_Year", "Bed_Room"])

center_lat = X["Latitude"].mean()
center_lon = X["Longitude"].mean()
R = 6371.0
lat1, lon1 = np.radians(center_lat), np.radians(center_lon)
lat2, lon2 = np.radians(X["Latitude"]), np.radians(X["Longitude"])
dlat = lat2 - lat1
dlon = lon2 - lon1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
X["Dist_to_Center"] = R * c
X = X.drop(columns=["Latitude", "Longitude"])
print_vif(X)

# %% Classification dataset
df_class = df.copy()
df_class["Price_Category"] = pd.qcut(df_class["SalePrice_10k"], q=3, labels=["Low", "Medium", "High"])
X_class = df_class.drop(columns=[
    "SalePrice_10k", "Price_Category", "Prop_Name_ENG", "ADDRESS_ENG",
    "Gross Area_Price", "SaleableAreaPrice", "Gross Area", "Estate_Size"
])
y_class = df_class["Price_Category"]

# %% Regression model training
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, confusion_matrix, accuracy_score
import scipy.stats as stats

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lin_model = LinearRegression()
kernel = (C(1.0, constant_value_bounds=(1e-3, 1e3))
          * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
          + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e1)))
gp_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
lin_cv_r2 = cross_val_score(lin_model, X_train_scaled, y_train, cv=kf, scoring="r2")
gp_cv_r2 = cross_val_score(gp_model, X_train_scaled, y_train, cv=kf, scoring="r2")
print("Linear Regression mean CV R2:", lin_cv_r2.mean())
print("Gaussian Process mean CV R2:", gp_cv_r2.mean())

lin_model.fit(X_train_scaled, y_train)
gp_model.fit(X_train_scaled, y_train)
y_pred_lin = lin_model.predict(X_test_scaled)
y_pred_gp = gp_model.predict(X_test_scaled)

results = pd.DataFrame({
    "Model": ["Linear Regression", "Gaussian Process Regression"],
    "CV Mean R2": [lin_cv_r2.mean(), gp_cv_r2.mean()],
    "Test R2": [r2_score(y_test, y_pred_lin), r2_score(y_test, y_pred_gp)],
    "Test RMSE": [np.sqrt(mean_squared_error(y_test, y_pred_lin)), np.sqrt(mean_squared_error(y_test, y_pred_gp))],
    "Test MAE": [mean_absolute_error(y_test, y_pred_lin), mean_absolute_error(y_test, y_pred_gp)]
}).sort_values(by="Test R2", ascending=False)
print(results)

plt.figure(figsize=(6, 4))
sns.barplot(data=results, x="Model", y="Test R2")
plt.title("Model Comparison by Test R2")
plt.show()

best_model_name = results.iloc[0]["Model"]
y_pred_best = y_pred_lin if best_model_name == "Linear Regression" else y_pred_gp

# %% Regression diagnostics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, pred, name in [(axes[0], y_pred_lin, "Linear Regression"), (axes[1], y_pred_gp, "Gaussian Process Regression")]:
    residuals = y_test - pred
    ax.scatter(pred, residuals, alpha=0.6, edgecolors="k")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title(f"{name}\nR² = {r2_score(y_test, pred):.3f}")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
stats.probplot(y_test - y_pred_lin, dist="norm", plot=axes[0])
axes[0].set_title("Linear Regression Q-Q Plot")
stats.probplot(y_test - y_pred_gp, dist="norm", plot=axes[1])
axes[1].set_title("Gaussian Process Q-Q Plot")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_best, alpha=0.7, edgecolors="k", s=30)
min_val = min(y_test.min(), y_pred_best.min())
max_val = max(y_test.max(), y_pred_best.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
plt.xlabel("Actual Price (10k HKD)")
plt.ylabel("Predicted Price (10k HKD)")
plt.title(f"Actual vs Predicted ({best_model_name})")
plt.legend()
plt.tight_layout()
plt.show()

# %% Feature importance
from sklearn.inspection import permutation_importance

lin_coefs = pd.Series(lin_model.coef_, index=X.columns)
print("Top linear coefficients:\n", lin_coefs.reindex(lin_coefs.abs().sort_values(ascending=False).index).head(5))
perm_imp = permutation_importance(gp_model, X_test_scaled, y_test, n_repeats=10, random_state=42)
perm_imp_series = pd.Series(perm_imp.importances_mean, index=X.columns)
print("Top GPR permutation importance:\n", perm_imp_series.sort_values(ascending=False).head(5))

# %% Classification model training
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
dt_model.fit(Xc_train, yc_train)
rf_model.fit(Xc_train, yc_train)
yc_pred_dt = dt_model.predict(Xc_test)
yc_pred_rf = rf_model.predict(Xc_test)

scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
cv_dt = cross_validate(dt_model, X_class, y_class, cv=5, scoring=scoring)
cv_rf = cross_validate(rf_model, X_class, y_class, cv=5, scoring=scoring)
for model_name, cv_res in [("Decision Tree", cv_dt), ("Random Forest", cv_rf)]:
    print(f"\n{model_name} five-fold CV")
    for metric in scoring:
        values = cv_res[f"test_{metric}"]
        print(f"{metric}: {values.mean():.4f} ± {values.std():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, pred, model, title, cmap in [
    (axes[0], yc_pred_dt, dt_model, "Decision Tree", "Blues"),
    (axes[1], yc_pred_rf, rf_model, "Random Forest", "Greens")
]:
    cm = confusion_matrix(yc_test, pred, labels=model.classes_, normalize="true")
    sns.heatmap(cm, annot=True, fmt=".2f", cmap=cmap, xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    ax.set_title(f"{title} Normalised Confusion Matrix")
plt.tight_layout()
plt.show()

rf_importance = pd.DataFrame({"Feature": X_class.columns, "Importance": rf_model.feature_importances_}).sort_values(by="Importance", ascending=False)
dt_importance = pd.DataFrame({"Feature": X_class.columns, "Importance": dt_model.feature_importances_}).sort_values(by="Importance", ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=rf_importance.head(10), x="Importance", y="Feature", ax=axes[0], color="steelblue")
axes[0].set_title("Random Forest Feature Importance")
sns.barplot(data=dt_importance.head(10), x="Importance", y="Feature", ax=axes[1], color="seagreen")
axes[1].set_title("Decision Tree Feature Importance")
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 8))
plot_tree(dt_model, feature_names=X_class.columns, class_names=dt_model.classes_, filled=True, max_depth=3, rounded=True)
plt.title("Decision Tree: Top Levels")
plt.show()

# %% Hypothesis testing
X_ols = sm.add_constant(X_train_scaled)
ols_model = sm.OLS(y_train, X_ols).fit()
print(ols_model.summary())
feature_names = ["const"] + list(X.columns)
for feature in ["SaleableArea", "Build_Ages"]:
    idx = feature_names.index(feature)
    print(feature, "coefficient:", ols_model.params[idx], "p-value:", ols_model.pvalues[idx])

from scipy.stats import ttest_rel, norm

t_stat, p_value = ttest_rel(gp_cv_r2, lin_cv_r2)
print("Paired CV comparison: t =", t_stat, "p =", p_value)

p0 = 1 / 3
for name, pred in [("Decision Tree", yc_pred_dt), ("Random Forest", yc_pred_rf)]:
    acc = accuracy_score(yc_test, pred)
    n = len(yc_test)
    z_stat = (acc - p0) / np.sqrt(p0 * (1 - p0) / n)
    p_value = 1 - norm.cdf(z_stat)
    print(f"{name}: accuracy={acc:.2%}, baseline z={z_stat:.2f}, p={p_value:.6f}")
