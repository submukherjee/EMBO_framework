import os
import numpy as np
import pandas as pd
import joblib
from datetime import timedelta
import matplotlib.pyplot as plt
import sys
import time

# Ensure Task2_3 module path is included
#sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasso_flowrate_module_trust import *

#### Requirements #### 

## need to do pip install scipy pandas scikit-learn matplotlib openpyxl joblib

### Then run
###cd "E:\Project_NAWI\Optimization_work\Task2_3_Surrogate"

# === GET INPUT PATH FROM COMMAND LINE ===
if len(sys.argv) < 2:
    raise ValueError("No input file provided. Usage: python run_optimization.py path/to/training_data.xlsx")


# === CONFIGURATION ===
### This can be changed according to directory path
data_file = os.path.abspath(sys.argv[1])  # Full path to the training data file e.g., C:/15830_mqtt_v2/2025-07-18_09-00-00_GMT/training_data.xlsx
main_directory = os.path.dirname(data_file)  # Timestamped input folder (created in Task 1/4) # e.g., C:/15830_mqtt_v2/2025-07-18_09-00-00_GMT
print(main_directory)
tariff_dir = os.path.join(os.path.abspath("C:/15830_mqtt_v2/"), "static_config") # Central tariff file location
output_dir = os.path.join(main_directory, "demo_results")  # Save results under current run folder e.g., C:/15830_mqtt_v2/2025-07-18_09-00-00_GMT/demo_results
os.makedirs(output_dir, exist_ok=True)

#data_file = "training_data_250626-030619.xlsx"

#data_file = "NAWI5.09_64Combinations.csv"
tariff_file = "tariff_test.xlsx"

flow_features = ["AI_10_FIT40", "AI_11_FIT50", "AI_12_FIT35"]  # Qc, Qp, Qr

# flow_features = [
#     "RO Overall Concentrate Flowrate (gpm)",
#     "RO Overall Permeate Flowrate (gpm)",
#     "Recirculation Flow Rate (gpm)"
# ]
dependent_feature = "OPTO22_RIO.True_Power"
#dependent_feature = "True Power (W)"
n_hours = 168
target_volume = 2.7 * n_hours * 60
recovery_target = 0.5125
hourly_recovery_limit = 0.8
shrinkage_percent = 0.05
smoothness_weight = 1e3
init_type = "physical"  # Can also use: "physical", "midpoint", or "hourly_median"
dt = 1.0

#### linear physical constraints derived from 3D convex hull
A = np.array([
    [0.95698817, -0.01655168,  0.28965444],
    [1.00000000,  0.00000000,  0.00000000],
    [-0.88298691, -0.08590144,  0.46147054],
    [0.80341152,  0.23730661,  0.54609111],
    [-0.60336339,  0.48717044,  0.63136169],
    [-0.72554232,  0.68286336, -0.08535792],
    [-0.83685216,  0.53254229, -0.12679578],
    [-0.84532472,  0.52647417, -0.09083533],
    [-0.83459832,  0.47930747, -0.27149584],
    [0.99536449, -0.03756092, -0.08853646],
    [0.93415396, -0.26258557,  0.24167167],
    [-0.00974596, -0.99586042,  0.09037168],
    [-0.01720291, -0.99776871,  0.06451091],
    [0.53575248, -0.14779379, -0.83134005],
    [0.24287070, -0.33394721, -0.91076511],
    [-0.60588120,  0.20635084, -0.76832760],
    [-0.71091275,  0.32811358, -0.62204866],
    [-0.15409965, -0.15409965, -0.97596444],
    [-0.54148350,  0.16921359, -0.82350615],
    [0.16883312, -0.26380174, -0.94968627],
    [0.29157512, -0.11059746, -0.95013270],
    [0.00000000,  0.00000000, -1.00000000],
    [-0.57735027, -0.57735027, -0.57735027]
])

b = np.array([
     4.11459770,  3.90000000, -0.23373183,  5.16985309,  1.85712731,  0.37984274,
    -0.76077469, -0.71834064, -1.00218835,  3.70538517,  3.61392100, -0.37211866,
    -0.41932090,  1.39480386,  0.15179419, -1.29781508, -1.23042592, -0.70885839,
    -1.21156932, -0.00211041,  0.61733490,  0.00000000, -1.73205081
])

perturb_bounds = np.array([
    [0.4, 4.9],   # conc
    [1.0, 3.9],   # perm
    [1e-3, 5.0]   # rcyc
])
lower_bounds = perturb_bounds[:, 0]
upper_bounds = perturb_bounds[:, 1]

# === LOAD DATA ===
data = pd.read_excel(data_file)
tariff_path = os.path.join(tariff_dir, tariff_file)

data["Datetime"] = pd.to_datetime(data["Datetime"])

#data["Datetime"] = pd.to_datetime(data["Date"] + " " + data["Time"], format="%m/%d/%Y %H:%M:%S")
data = data.sort_values("Datetime")
data = data.dropna(subset=flow_features + [dependent_feature])

# === Tariff Alignment ===
tariff_vector = pd.read_excel(tariff_path).iloc[:, 1].values
hourly_tariff = align_tariff_to_experiment(tariff_vector, data["Datetime"].max(), n_hours)

# === TRAIN LASSO MODEL ===
X_train = data[flow_features]
y_train = data[dependent_feature]
if len(data) < 500:
    model = LassoSurrogateModel(poly_order=2)
else:
    model = LassoSurrogateModel(poly_order=3)

model.train(X_train, y_train)

# === OPTIMIZATION ===
opt_result = optimize_flowrates_lasso(
    model=model,
    hourly_tariff=hourly_tariff,
    n_hours=n_hours,
    flow_features=flow_features,
    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds,
    A=A,
    b=b,
    target_volume=target_volume,
    recovery_target=recovery_target,
    hourly_recovery_limit=hourly_recovery_limit,
    smooth_weight=smoothness_weight,
    dt=dt,
    shrinkage_percent=shrinkage_percent,
    init_type=init_type,
    data=data
)

optimized_df, cost, success, message, constraint_check, trained_model, cost_history = opt_result

if not success and "time limit" in str(message).lower():
    print("[WARNING] Optimization terminated early due to 20-minute timeout. Results are best-so-far.")

# === SAVE OUTPUTS ===
# === Save Directory Setup ===
start_time = data["Datetime"].min().strftime("%Y%m%d_%H%M")
end_time = data["Datetime"].max().strftime("%Y%m%d_%H%M")
cycle_dir = os.path.join(output_dir, f"Cycle_{start_time}_to_{end_time}")
os.makedirs(cycle_dir, exist_ok=True)

future_dates = pd.date_range(start=(data["Datetime"].max() + timedelta(hours=1)).ceil("h"), periods=n_hours, freq="h")

### Say if the last observation in the input data file ends at 8:33 AM then instead of 10 AM consideration, if want to have at 9 AM (that means always at next hour)

##  future_dates = pd.date_range(start=data["Datetime"].max().ceil("H"),periods=n_hours,freq="H")

### If it ends at 8:05 AM, then 9 AM will be considered, if it ends at 8:33 then 10 AM will be considered
# future_dates = pd.date_range(
#     start=(data["Datetime"].max() + timedelta(hours=1, minutes=30)).floor("H"),
#     periods=n_hours, freq="H"
# )

optimized_df["Datetime"] = future_dates

# === Save Optimized Setpoints ===
optimized_df.to_excel(os.path.join(cycle_dir, "optimized_setpoints.xlsx"), index=False)

# === PREDICTION ===
#predicted_power = trained_model.predict(optimized_df[flow_features])

predicted_power = np.clip(trained_model.predict(optimized_df[flow_features]), 0, None) # prevent negative power
# === Save Predicted Power ===
pd.DataFrame({"Hour": range(1, n_hours + 1), "Predicted_Power": predicted_power}).to_excel(
    os.path.join(cycle_dir, "predicted_power.xlsx"), index=False
)

# === COEFFICIENTS ===
# === Save LASSO Coefficients ===
lasso_coef = trained_model.coef_
feature_names = trained_model.poly.get_feature_names_out(input_features=flow_features)
pd.DataFrame({"Feature": feature_names, "Coefficient": lasso_coef}).to_excel(
    os.path.join(cycle_dir, "lasso_coefficients.xlsx"), index=False
)

# Save the entire model object
joblib.dump(trained_model, os.path.join(cycle_dir, "lasso_model_info.pkl"))

# Save LASSO diagnostics (lambda values, MSEs, nonzero count)
diagnostics = pd.DataFrame({
    "lambda_min": [trained_model.alpha_min],
    "lambda_1se": [trained_model.alpha_1se],
    "lambda_opt": [trained_model.alpha_opt],
    "selected_lambda": [trained_model.alpha_opt],
    "mse_at_lambda_min": [trained_model.mse_at_min],
    "mse_at_lambda_1se": [trained_model.mse_at_1se],
    "mse_at_lambda_opt": [trained_model.mse_at_opt],
    "nonzero_coefficients": [trained_model.nonzero_at_lambda_opt],
    "lambda_opt_reason": [trained_model.lambda_opt_label]
})

diagnostics.to_excel(os.path.join(cycle_dir, "lasso_lambda_diagnostics.xlsx"), index=False)

### To see the predictions later

#loaded_model = joblib.load(os.path.join(cycle_dir, "lasso_model_info.pkl"))
#predictions = loaded_model.predict(new_input_df)


# === COST ===
pd.DataFrame({"optimized_cost": [cost]}).to_excel(os.path.join(cycle_dir, "optimized_cost.xlsx"), index=False)

# === Save Convergence Status ===
pd.DataFrame({
    "Success": [success],
    "Message": [message]
}).to_excel(os.path.join(cycle_dir, "convergence_status.xlsx"), index=False)


# === CONSTRAINT CHECK ===
pd.DataFrame({"Constraint": list(constraint_check.keys()), "Satisfied": list(constraint_check.values())}).to_excel(
    os.path.join(cycle_dir, "constraint_validation.xlsx"), index=False
)

perturb_seed = int(time.time()) % (2**32 - 1)

print(f"[INFO] Using perturbation seed: {perturb_seed}")

with open(os.path.join(cycle_dir, "perturbation_seed.txt"), "w") as f:

    f.write(f"Perturbation seed used: {perturb_seed}\n")


# === PERTURBED SETPOINTS ===
perturbed_df = generate_perturbed_setpoints(
    optimized_df, flow_features, lower_bounds, upper_bounds, shrinkage_percent=shrinkage_percent, perturbation_fraction= 0.98, seed=perturb_seed
)
perturbed_df["Datetime"] = future_dates
perturbed_df.to_excel(os.path.join(cycle_dir, "optimized_setpoints_perturbed.xlsx"), index=False)

# Save rounded perturbed setpoints

rounded_path = os.path.join(cycle_dir, "optimized_setpoints_perturbed_rounded.xlsx")
save_rounded_perturbed_setpoints(perturbed_df, flow_features, rounded_path)

#### Save a copy of rounded setpoints by dropping Datetime column in main directory

rounded_df = pd.read_excel(rounded_path)

# Drop the 'Datetime' column if you only want to keep flowrate setpoints
if "Datetime" in rounded_df.columns:
    rounded_df = rounded_df.drop(columns=["Datetime"])

updated_setpoints_path = os.path.join(main_directory, "updated_setpoints.xlsx")
rounded_df.to_excel(updated_setpoints_path, index=False)

### If only one row is required

# Load the rounded perturbed file
# rounded_df = pd.read_excel(os.path.join(cycle_dir, "optimized_setpoints_perturbed_rounded.xlsx"))

# # Drop Datetime column if it exists
# if "Datetime" in rounded_df.columns:
#     rounded_df = rounded_df.drop(columns=["Datetime"])

# # Extract first row only
# first_row = rounded_df.iloc[[0]]

# # Save to updated_setpoints.xlsx in main_directory (for Task 4)
# updated_setpoints_path = os.path.join(main_directory, "updated_setpoints.xlsx")
# first_row.to_excel(updated_setpoints_path, index=False)


# === PLOTS ===
power_initial = np.clip(trained_model.predict(generate_initial_setpoints(
    init_type=init_type,
    n_hours=n_hours,
    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds,
    shrinkage_percent=shrinkage_percent,
    flow_features=flow_features,
    data=data,
    target_volume=target_volume,
    recovery_target=recovery_target
)),0,None) # prevent negative power
cost_initial = power_initial / 1000 * hourly_tariff
cost_optimized = predicted_power / 1000 * hourly_tariff

# === Save Cost Comparison Summary ===
total_initial_cost = np.sum(cost_initial)
total_optimized_cost = np.sum(cost_optimized)
cost_savings = total_initial_cost - total_optimized_cost
reduction_pct = 100 * cost_savings / total_initial_cost

summary_df = pd.DataFrame({
    "Total Initial Cost ($)": [round(total_initial_cost, 2)],
    "Total Optimized Cost ($)": [round(total_optimized_cost, 2)],
    "Cost Savings ($)": [round(cost_savings, 2)],
    "Cost Reduction (%)": [round(reduction_pct, 2)]
})

summary_df.to_csv(os.path.join(cycle_dir, "cost_comparison_summary.csv"), index=False)

# Cost Comparison Plot
plt.figure(figsize=(10, 5))
plt.plot(range(n_hours), cost_initial, label="Initial", linewidth=2)
plt.plot(range(n_hours), cost_optimized, label="Optimized", linewidth=2)
plt.title("Hourly Cost: Initial vs Optimized")
plt.xlabel("Hour")
plt.ylabel("Cost ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(cycle_dir, "initial_optimized_cost.png"))
plt.close()

# Power Comparison Plot
plt.figure(figsize=(10, 5))
plt.plot(range(n_hours), power_initial, label="Initial", linewidth=2)
plt.plot(range(n_hours), predicted_power, label="Optimized", linewidth=2)
plt.title("Predicted Power: Initial vs Optimized")
plt.xlabel("Hour")
plt.ylabel("Power (kW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(cycle_dir, "initial_optimized_power.png"))
plt.close()



plt.figure(figsize=(10, 5))
plt.plot(cost_history, marker='o')
plt.title("Optimization Convergence: Cost vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("Objective Cost")
plt.ticklabel_format(style='plain', axis='y')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(cycle_dir, "optimization_convergence.png"))
plt.close()

# Normalized Power vs Tariff
power_norm = (predicted_power - predicted_power.min()) / (predicted_power.max() - predicted_power.min())
tariff_norm = (hourly_tariff - hourly_tariff.min()) / (hourly_tariff.max() - hourly_tariff.min())

plt.figure(figsize=(10, 5))
plt.plot(range(n_hours), power_norm, label="Normalized Power", linewidth=2)
plt.plot(range(n_hours), tariff_norm, label="Normalized Tariff", linestyle="--", linewidth=2)
plt.title("Normalized Power vs Tariff")
plt.xlabel("Hour")
plt.ylabel("Normalized Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(cycle_dir, "normalized_power_tariff_plot.png"))
plt.close()

# Plot optimized vs initial flowrates for each variable
initial_df = generate_initial_setpoints(
    init_type=init_type,
    n_hours=n_hours,
    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds,
    shrinkage_percent=shrinkage_percent,
    flow_features=flow_features,
    data=data,
    target_volume=target_volume,
    recovery_target=recovery_target
)

for feature in flow_features:
    plt.figure(figsize=(10, 5))
    plt.plot(range(n_hours), initial_df[feature], label="Initial", color='gray', linewidth=2)
    plt.plot(range(n_hours), optimized_df[feature], label="Optimized", color='blue', linewidth=2)
    plt.title(f"{feature}: Initial vs Optimized")
    plt.xlabel("Hour")
    plt.ylabel("Flowrate (gpm)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"{feature.replace('/', '_')}_initial_vs_optimized.png"
    plt.savefig(os.path.join(cycle_dir, filename))
    plt.close()


for feature in flow_features:
    plt.figure(figsize=(10, 5))
    plt.plot(range(n_hours), initial_df[feature], label="Initial", color='gray', linewidth=2)
    plt.plot(range(n_hours), perturbed_df[feature], label="Optimized", color='blue', linewidth=2)
    plt.title(f"{feature}: Initial vs Optimized")
    plt.xlabel("Hour")
    plt.ylabel("Flowrate (gpm)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"{feature.replace('/', '_')}_initial_vs_optimized_perturbed.png"
    plt.savefig(os.path.join(cycle_dir, filename))
    plt.close()

print(f"\nOptimization complete. Results saved to: {cycle_dir}")
