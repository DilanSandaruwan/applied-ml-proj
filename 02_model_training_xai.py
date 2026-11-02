import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
import shap

# 1. Load Dataset
df = pd.read_csv("colombo_air_quality.csv")
features = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "precipitation",
    "hour", "day_of_week", "is_weekend", "pm25_lag_1h"
]
target = "pm25"

X = df[features].copy()
y = df[target].copy()

# 2. Chronological Split (70% Train, 15% Validation, 15% Test)
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

# 3. Standard Scaling (Mandatory for distance-based covariance kernels)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 4. Fit Baseline Model (from lectures: Ridge Regression)
baseline = Ridge(alpha=1.0)
baseline.fit(X_train_scaled, y_train)
y_pred_base = baseline.predict(X_test_scaled)

# 5. Fit Novel Algorithm: Gaussian Process Regressor (GPR)
np.random.seed(42)
gpr_idx = np.random.choice(len(X_train_scaled), size=1200, replace=False)
X_gpr_train = X_train_scaled[gpr_idx]
y_gpr_train = y_train.iloc[gpr_idx]

kernel = Matern(length_scale=np.ones(X.shape[1]), nu=2.5) + WhiteKernel(noise_level=1.0)
gpr = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=5,
    normalize_y=True,
    random_state=42
)

print("Optimizing GPR kernel hyperparameters via Marginal Log-Likelihood...")
gpr.fit(X_gpr_train, y_gpr_train)
print(f"Optimized Kernel: {gpr.kernel_}")

# 6. Performance Evaluation and Prediction Interval Coverage Probability (PICP)
y_pred_gpr, y_std_gpr = gpr.predict(X_test_scaled, return_std=True)

lower_bound = y_pred_gpr - 1.96 * y_std_gpr
upper_bound = y_pred_gpr + 1.96 * y_std_gpr
picp = np.mean((y_test.values >= lower_bound) & (y_test.values <= upper_bound)) * 100

def compute_metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

metrics_base = compute_metrics(y_test, y_pred_base)
metrics_gpr = compute_metrics(y_test, y_pred_gpr)

print("\n--- MODEL PERFORMANCE COMPARISON ---")
print(f"Baseline (Ridge): RMSE={metrics_base['RMSE']:.3f}, MAE={metrics_base['MAE']:.3f}, R2={metrics_base['R2']:.3f}")
print(f"Novel (GPR):     RMSE={metrics_gpr['RMSE']:.3f}, MAE={metrics_gpr['MAE']:.3f}, R2={metrics_gpr['R2']:.3f}")
print(f"GPR 95% Uncertainty Interval Coverage (PICP): {picp:.2f}%\n")

# 7. Explainable AI: SHAP Kernel Explainer
print("Computing SHAP explanations using KernelExplainer...")
background_data = shap.kmeans(X_gpr_train, 25)
explainer = shap.KernelExplainer(gpr.predict, background_data)
shap_sample = X_test_scaled[:60]
shap_values = explainer.shap_values(shap_sample)

# Export SHAP Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, shap_sample, feature_names=features, show=False)
plt.title("SHAP Feature Importance (Gaussian Process Model)", fontsize=13)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300)
plt.close()

# 8. Explainable AI: Partial Dependence Plots (PDP)
fig, ax = plt.subplots(figsize=(12, 4))
PartialDependenceDisplay.from_estimator(
    gpr, X_train_scaled, 
    features=[features.index("wind_speed_10m"), features.index("pm25_lag_1h")],
    feature_names=features,
    ax=ax
)
plt.suptitle("Partial Dependence Plots (PDP) - Atmospheric Drivers", fontsize=13)
plt.tight_layout()
plt.savefig("pdp_plots.png", dpi=300)
plt.close()

# 9. Save Artifacts for Streamlit Frontend
os.makedirs("models", exist_ok=True)
joblib.dump(gpr, "models/gpr_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(features, "models/features.pkl")
joblib.dump(explainer, "models/shap_explainer.pkl")
print("All artifacts and plots exported successfully.")