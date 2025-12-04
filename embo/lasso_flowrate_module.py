import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LassoCV, Lasso, lasso_path, LinearRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import KFold, TimeSeriesSplit
from scipy.optimize import minimize, Bounds, LinearConstraint
import pickle
import warnings
import pytz
import time

def compute_total_cost(predicted_power, hourly_tariff):
    """
    Computes the total energy cost.

    Args:
        predicted_power (np.ndarray): Predicted power values in watts.
        hourly_tariff (np.ndarray): Tariff per kWh for each hour.

    Returns:
        float: Total cost in currency unit.
    
    """
    return np.sum(predicted_power / 1000 * hourly_tariff)

def smoothness_penalty(x, n_hours, n_vars):
    """Compute smoothness penalty for optimization objective."""
    return sum(np.sum(np.diff(x[i * n_hours:(i + 1) * n_hours]) ** 2) for i in range(n_vars))


class LassoSurrogateModel:
    """
    Sparse polynomial LASSO surrogate model with custom lambda selection and stepwise fallback.
    """

    def __init__(self, poly_order=3):
        self.poly_order = poly_order
        self.scaler = None
        self.poly = None
        self.true_power_mean = None
        self.true_power_std = None
        self.feature_names = None

        # Lambda-related
        self.lambda_seq = np.power(10.0, np.arange(2, -4.01, -0.1))
        self.alpha_min = None
        self.alpha_1se = None
        self.alpha_opt = None
        self.lambda_opt_label = None
        self.mse_at_min = None
        self.mse_at_1se = None
        self.mse_at_opt = None
        self.nonzero_at_lambda_opt = None

        # Model coefficients
        self.coef_ = None
        self.intercept_ = None

    def train(self, X_raw, y_raw):
        """
        Trains the LASSO model with custom lambda selection and fallback to forward stepwise regression.
        """
        if X_raw.shape[0] < 300 and self.poly_order > 2:
            print(f"[WARN] Reducing poly_order from {self.poly_order} to 2 due to small training size.")
            self.poly_order = 2

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw)
        self.feature_names = X_raw.columns.tolist()
        self.true_power_mean = y_raw.mean()
        self.true_power_std = y_raw.std()
        y_scaled = (y_raw - self.true_power_mean) / self.true_power_std

        self.poly = PolynomialFeatures(degree=self.poly_order, include_bias=False)
        X_poly = self.poly.fit_transform(X_scaled)

        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        print(f"[INFO] Fitting LassoCV with {cv.get_n_splits()} folds and {len(self.lambda_seq)} lambdas...")
        start_time = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso_cv = LassoCV(alphas=self.lambda_seq, cv=cv, max_iter=100000, selection='random').fit(X_poly, y_scaled)
        print(f"[INFO] LassoCV fitting completed in {time.time() - start_time:.2f} seconds.")

        self.alpha_min = lasso_cv.alpha_
        mean_mse = np.mean(lasso_cv.mse_path_, axis=1)
        std_mse = np.std(lasso_cv.mse_path_, axis=1)

        idx_min = np.argmin(mean_mse)
        threshold = mean_mse[idx_min] + std_mse[idx_min]
        idx_1se = np.min(np.where(mean_mse <= threshold)[0])

        self.alpha_1se = self.lambda_seq[idx_1se]
        self.mse_at_min = mean_mse[idx_min]
        self.mse_at_1se = mean_mse[idx_1se]

        try:
            coef_path = lasso_cv.coef_path_
        except AttributeError:
            print("[WARN] lasso_cv.coef_path_ not available — using fallback")
            _, coef_path, _ = lasso_path(X_poly, y_scaled, alphas=self.lambda_seq)

        alphas = lasso_cv.alphas_
        nonzero_counts = np.sum(coef_path != 0, axis=0)

        idx_min_lam = np.where(alphas == self.alpha_min)[0][0]
        idx_1se_lam = np.where(alphas == self.alpha_1se)[0][0]
        nonzero_min = nonzero_counts[idx_min_lam]

        if nonzero_min <= 8:
            idx_opt = idx_min_lam
            self.lambda_opt_label = "lambda_min (≤ 8 nonzero)"
        else:
            valid_range = [
                i for i in range(len(alphas) - 1, -1, -1)
                if alphas[i] >= self.alpha_min and alphas[i] <= self.alpha_1se
            ]
            sparse_candidates = [i for i in valid_range if nonzero_counts[i] <= 7]
            if sparse_candidates:
                idx_opt = sparse_candidates[0]
                self.lambda_opt_label = "first λ in [min,1se] with ≤ 7 nonzero"
            else:
                idx_opt = idx_1se_lam
                self.lambda_opt_label = "fallback: lambda_1se"

        self.alpha_opt = alphas[idx_opt]
        self.mse_at_opt = mean_mse[idx_opt]
        self.nonzero_at_lambda_opt = nonzero_counts[idx_opt]

        # === Fallback #1: lambda_min gives 0 nonzero coefficients
        if self.nonzero_at_lambda_opt == 0:
            print("[WARN] Selected lambda resulted in 0 nonzero coefficients. Falling back to lambda_min.")
            idx_opt = idx_min_lam
            self.alpha_opt = alphas[idx_opt]
            self.mse_at_opt = mean_mse[idx_opt]
            self.nonzero_at_lambda_opt = nonzero_counts[idx_opt]
            self.lambda_opt_label = "fallback: lambda_min (forced, nonzero=0)"

        self.coef_ = coef_path[:, idx_opt]
        self.intercept_ = np.mean(y_scaled - X_poly @ self.coef_)

        # === Fallback #2: If still zero coefficients, apply forward stepwise regression on raw inputs
        if self.nonzero_at_lambda_opt == 0:
            print("[CRITICAL] lambda_min also gave 0 nonzero coefficients. Applying forward stepwise regression...")
            from sklearn.metrics import mean_squared_error

            # Step 1: First best single feature
            min_mse = np.inf
            best_single = None
            for col in X_raw.columns:
                X_one = X_raw[[col]]
                model = LinearRegression().fit(X_one, y_raw)
                mse = mean_squared_error(y_raw, model.predict(X_one))
                if mse < min_mse:
                    min_mse = mse
                    best_single = col

            # Step 2: Add second feature
            remaining = [f for f in X_raw.columns if f != best_single]
            best_pair = [best_single]
            min_pair_mse = min_mse
            for col in remaining:
                pair = X_raw[[best_single, col]]
                model = LinearRegression().fit(pair, y_raw)
                mse = mean_squared_error(y_raw, model.predict(pair))
                if mse < min_pair_mse:
                    min_pair_mse = mse
                    best_pair = [best_single, col]

            print(f"[INFO] Forward stepwise selected features: {best_pair}")

            # Fit final linear model
            X_top = X_raw[best_pair]
            X_top_scaled = self.scaler.transform(X_raw)[..., [X_raw.columns.get_loc(f) for f in best_pair]]
            linreg = LinearRegression().fit(X_top_scaled, y_scaled)

            # Match polynomial feature space
            self.poly.fit(X_scaled)
            feature_names = self.poly.get_feature_names_out(self.feature_names)
            X_poly_all = self.poly.transform(X_scaled)
            self.coef_ = np.zeros(X_poly_all.shape[1])

            for i, name in enumerate(feature_names):
                for j, raw_name in enumerate(best_pair):
                    if name == raw_name:
                        self.coef_[i] = linreg.coef_[j]

            self.intercept_ = linreg.intercept_
            self.alpha_opt = 0.0
            self.lambda_opt_label = "Manual fallback: top-2 via forward stepwise"
            self.nonzero_at_lambda_opt = 2

        # === Logs ===
        print(f"[INFO] lambda_min = {self.alpha_min:.5e}, MSE = {self.mse_at_min:.5f}")
        print(f"[INFO] lambda_1se = {self.alpha_1se:.5e}, MSE = {self.mse_at_1se:.5f}")
        print(f"[INFO] lambda_opt = {self.alpha_opt:.5e} ({self.lambda_opt_label})")
        print(f"[INFO] MSE at lambda_opt = {self.mse_at_opt:.5f}")
        print(f"[INFO] Nonzero Coefficients at lambda_opt = {self.nonzero_at_lambda_opt}")

    def predict(self, X_raw):
        X_scaled = self.scaler.transform(X_raw)
        X_poly = self.poly.transform(X_scaled)
        y_scaled_pred = X_poly @ self.coef_ + self.intercept_
        return y_scaled_pred * self.true_power_std + self.true_power_mean


def generate_initial_setpoints(init_type, n_hours, lower_bounds, upper_bounds,
                               shrinkage_percent=0.05, flow_features=None,
                               data=None, target_volume=None, recovery_target=None):
    """
    Generate initial flowrate setpoints for optimization, using various initialization strategies.

    Args:
        init_type (str): Initialization method. One of:
            - 'midpoint': Uses the midpoint between shrunk bounds for each variable.
            - 'random': Samples uniformly within shrunk bounds.
            - 'hourly_median': Uses historical hourly medians from past data (requires `data`).
            - 'physical': Uses physically-derived initialization based on volume and recovery targets.
        n_hours (int): Number of future hours to generate (time horizon).
        lower_bounds (np.ndarray): Lower bounds for each flowrate (shape = [n_features]).
        upper_bounds (np.ndarray): Upper bounds for each flowrate (shape = [n_features]).
        shrinkage_percent (float, optional): Fraction by which to shrink the bounds. Default is 0.05.
        flow_features (list of str, optional): List of flowrate feature names. Required for column labeling.
        data (pd.DataFrame, optional): Past data used for 'hourly_median' strategy. Must include 'Datetime' and flow_features.
        target_volume (float, optional): Total desired permeate volume for 'physical' strategy.
        recovery_target (float, optional): Recovery ratio used to infer concentrate flow in 'physical' strategy.

    Returns:
        pd.DataFrame: Initial flowrate setpoints with shape (n_hours, n_features).
                      Columns correspond to `flow_features`.
    """
    range_bounds = upper_bounds - lower_bounds
    lb_shrunk = lower_bounds + shrinkage_percent * range_bounds
    ub_shrunk = upper_bounds - shrinkage_percent * range_bounds

    if init_type == "midpoint":
        init_vals = np.array([np.full(n_hours, (lb + ub) / 2) for lb, ub in zip(lb_shrunk, ub_shrunk)])
    elif init_type == "random":
        np.random.seed(42)
        init_vals = np.array([np.random.uniform(lb, ub, n_hours) for lb, ub in zip(lb_shrunk, ub_shrunk)])
    elif init_type == "hourly_median" and data is not None:
        grouped = data.copy()
        grouped["Datetime"] = pd.to_datetime(grouped["Datetime"])
        hourly = grouped.groupby(grouped["Datetime"].dt.floor("h"))[flow_features].median().reset_index()
        hourly = hourly.iloc[:n_hours]
        if len(hourly) < n_hours:
            last_row = hourly.iloc[-1:].copy()
            hourly = pd.concat([hourly] + [last_row] * (n_hours - len(hourly)), ignore_index=True)
        return hourly[flow_features].copy()
    elif init_type == "physical":
        Qp = target_volume / (n_hours*60)
        Qc = ((1 - recovery_target) / recovery_target) * Qp
        Qr = 0.69
        Qp = np.clip(Qp, lb_shrunk[1], ub_shrunk[1])
        Qc = np.clip(Qc, lb_shrunk[0], ub_shrunk[0])
        Qr = np.clip(Qr, lb_shrunk[2], ub_shrunk[2])
        init_vals = np.tile([Qc, Qp, Qr], (n_hours, 1)).T
    else:
        raise ValueError("Invalid init_type or missing data for hourly_median.")

    return pd.DataFrame(init_vals.T, columns=flow_features)

def generate_perturbed_setpoints(df, flow_features, lower_bounds, upper_bounds,
                                  shrinkage_percent=0.05, perturbation_fraction=0.95, seed=42):
    """
    Generate randomly perturbed setpoints within shrunk convex hull bounds.
    """
    np.random.seed(seed)
    range_ = upper_bounds - lower_bounds
    perturb_range = range_ * shrinkage_percent * perturbation_fraction
    perturb = np.random.uniform(-1, 1, size=(len(df), len(flow_features))) * perturb_range
    return pd.DataFrame(df[flow_features].values + perturb, columns=flow_features)


def generate_perturbed_setpoints_1(df, flow_features, lower_bounds, upper_bounds,
                                  perturbation_fraction=0.05, seed=42):
    """
    Generate small random perturbations around optimized setpoints,
    ensuring perturbed values always remain within physical bounds.
    """
    np.random.seed(seed)
    Q_opt = df[flow_features].values
    Q_min = lower_bounds.reshape(1, -1)
    Q_max = upper_bounds.reshape(1, -1)

    # Compute per-element safe perturbation ranges
    delta = np.minimum(Q_opt - Q_min, Q_max - Q_opt) * perturbation_fraction

    # Uniform perturbation scaled by range
    perturb = np.random.uniform(-1, 1, size=Q_opt.shape) * delta

    return pd.DataFrame(Q_opt + perturb, columns=flow_features)

def align_tariff_to_experiment(tariff_vector, end_time, n_hours):
    """
    Aligns tariff vector to start 1 hour after experiment ends, 
    rounding up to the nearest hour (equivalent to R's ceiling_date).

    Args:
        tariff_vector (np.ndarray): 168-length base tariff vector (7×24).
        end_time (pd.Timestamp): Timestamp of last data entry.
        n_hours (int): Number of future hours to predict.

    Returns:
        np.ndarray: Tariff vector aligned to future window.

    """
    start_time = (end_time + pd.Timedelta(hours=1)).ceil("h")
    # start_time = end_time.ceil("H")  # Removed +1 hour   ## If we need to remove 1 hour
    #ts = end_time + pd.Timedelta(hours=1)
    #start_time = (ts + pd.Timedelta(minutes=30)).floor("H")
    weekday_idx = start_time.dayofweek
    hour_of_day = start_time.hour
    start_index = weekday_idx * 24 + hour_of_day
    tariff_cyclic = np.resize(tariff_vector, start_index + n_hours)
    return tariff_cyclic[start_index:start_index + n_hours]

def save_rounded_perturbed_setpoints(perturbed_df, flow_features, save_path):
    """
    Round perturbed flowrate columns to 2 decimal places and save to a new Excel file.

    Args:
        perturbed_df (pd.DataFrame): DataFrame with original perturbed setpoints and Datetime.
        flow_features (list): List of flowrate column names to round.
        save_path (str): Output file path for saving the rounded DataFrame.
    """
    rounded_df = perturbed_df.copy()
    rounded_df[flow_features] = rounded_df[flow_features].round(2)
    rounded_df.to_excel(save_path, index=False)



def optimize_flowrates_lasso(model, hourly_tariff, n_hours, flow_features, lower_bounds,
                             upper_bounds, A, b, target_volume, recovery_target,
                             hourly_recovery_limit=0.8, smooth_weight=1e3, dt=1.0,
                             shrinkage_percent=0.05, init_type="physical", data=None):
    
    """
    Optimize flowrates using a LASSO surrogate model subject to safety and physical constraints.

    Args:
        model (LassoSurrogateModel): Trained LASSO surrogate model for power prediction.
        hourly_tariff (np.ndarray): Array of hourly electricity tariff values (length = n_hours).
        n_hours (int): Number of forecast hours (typically 168 for 7 days).
        flow_features (list of str): Names of flowrate features [e.g., "AI_10_FIT40", "AI_11_FIT50", "AI_12_FIT35"].
        lower_bounds (np.ndarray): Lower bounds for each flowrate (shape = [n_features]).
        upper_bounds (np.ndarray): Upper bounds for each flowrate (shape = [n_features]).
        A (np.ndarray): Matrix defining linear inequality constraints (Ax ≤ b), shape = [n_constraints, 3].
        b (np.ndarray): Vector defining bounds in linear inequality constraints, shape = [n_constraints].
        target_volume (float): Weekly cumulative permeate volume target.
        recovery_target (float): Desired average recovery ratio over the horizon.
        hourly_recovery_limit (float, optional): Max allowable recovery ratio per hour. Default is 0.8.
        smooth_weight (float, optional): Weight on smoothness regularization. Default is 1e3.
        dt (float, optional): Time interval per step (in hours). Default is 1.0.
        shrinkage_percent (float, optional): Fraction by which to shrink convex hull bounds. Default is 0.05.
        init_type (str, optional): Initialization strategy: 'physical', 'random', 'midpoint', or 'hourly_median'.
        data (pd.DataFrame, optional): Past data (required if init_type = 'hourly_median').

    Returns:
        df_out (pd.DataFrame): Optimized flowrate setpoints (shape = [n_hours, n_features]).
        objective (float): Final value of the optimization cost function.
        success (bool): Whether the optimizer terminated successfully.
        message (str): Termination reason provided by the optimizer.
        constraints_satisfied (dict): Dictionary indicating satisfaction of key constraints:
            - 'cumulative_volume'
            - 'recovery_ratio'
            - 'feed_flow_bounds_ok'
            - 'recovery_upper_ok'
            - 'recovery_linearized_ok'
            - 'Axb_ok'
        model (LassoSurrogateModel): The surrogate model used for optimization (for reuse).
    """

    range_bounds = upper_bounds - lower_bounds
    lb_shrunk = lower_bounds + shrinkage_percent * range_bounds
    ub_shrunk = upper_bounds - shrinkage_percent * range_bounds

    init_df = generate_initial_setpoints(init_type, n_hours, lb_shrunk, ub_shrunk,
                                         shrinkage_percent, flow_features, data,
                                         target_volume, recovery_target)
    x0 = init_df.values.flatten()
    cost_history = []

    # === Timing setup ===
    time_limit_sec = 20 * 60  # 20 minutes
    start_time = time.time()
    best_x = [x0]
    best_cost = [np.inf]

    def objective(x):
        df = pd.DataFrame(x.reshape(n_hours, len(flow_features)), columns=flow_features)
        pred = model.predict(df)
        pred = np.clip(pred, 0, None)
        cost = compute_total_cost(pred, hourly_tariff)
        return cost

    def callback(xk, state):
        elapsed = time.time() - start_time
        df = pd.DataFrame(xk.reshape(n_hours, len(flow_features)), columns=flow_features)
        pred = model.predict(df)
        pred = np.clip(pred, 0, None)
        cost = compute_total_cost(pred, hourly_tariff)
        cost_history.append(cost)

        # Save best so far
        if cost < best_cost[0]:
            best_cost[0] = cost
            best_x[0] = xk.copy()

        if elapsed > time_limit_sec:
            print(f"[TIMEOUT] Optimization exceeded {time_limit_sec / 60:.1f} minutes. Exiting early.")
            raise TimeoutError  # Will be caught below

    def eq_constraints(x):
        x_mat = x.reshape(n_hours, len(flow_features))
        Qc, Qp = x_mat[:, 0], x_mat[:, 1]
        return [
            np.sum(Qp) * dt * 60 - target_volume,
            (1 - recovery_target) * np.sum(Qp) - recovery_target * np.sum(Qc)
        ]

    def ineq_constraints(x):
        x_mat = x.reshape(n_hours, len(flow_features))
        Qc, Qp, Qr = x_mat[:, 0], x_mat[:, 1], x_mat[:, 2]

        flow_balance = np.concatenate([3 - (Qc + Qp + Qr), (Qc + Qp + Qr) - 16])
        recovery_hourly = (1 - hourly_recovery_limit) * Qp - hourly_recovery_limit * (Qc + Qr)
        recovery_linear = Qp - 4 * (Qc + Qr)
        Axb = np.concatenate([Ai[0]*Qp + Ai[1]*Qc + Ai[2]*Qr - bi for Ai, bi in zip(A, b)])
        return np.concatenate([flow_balance, recovery_hourly, recovery_linear, Axb])

    constraints = [
        {'type': 'eq', 'fun': eq_constraints},
        {'type': 'ineq', 'fun': lambda x: -ineq_constraints(x)}
    ]

    try:
        result = minimize(
            objective,
            x0,
            method='trust-constr',
            constraints=constraints,
            callback=callback,
            options={
                'disp': True,
                'maxiter': 300,
                'xtol': 1e-2,
                'gtol': 1e-2,
                'finite_diff_rel_step': 1e-2
            }
        )
        final_x = result.x
        success = result.success
        message = result.message
        final_cost = result.fun

    except TimeoutError:
        # Optimization was interrupted by the 20-minute timeout
        final_x = best_x[0]
        final_cost = best_cost[0]
        success = False
        message = f"Terminated after {time_limit_sec/60:.1f} minutes due to time limit."

    # === Build output DataFrame ===
    df_out = pd.DataFrame(final_x.reshape(n_hours, len(flow_features)), columns=flow_features)
    Qc, Qp, Qr = df_out[flow_features[0]], df_out[flow_features[1]], df_out[flow_features[2]]

    constraints_satisfied = {
        "cumulative_volume": abs(np.sum(Qp) * dt * 60 - target_volume) < 1e-3,
        "recovery_ratio": abs((1 - recovery_target) * np.sum(Qp) - recovery_target * np.sum(Qc)) < 1e-3,
        "feed_flow_bounds_ok": np.all((Qc + Qp + Qr >= 3) & (Qc + Qp + Qr <= 16)),
        "recovery_upper_ok": np.all((1 - hourly_recovery_limit) * Qp - hourly_recovery_limit * (Qc + Qr) <= 1e-5),
        "recovery_linearized_ok": np.all(Qp - 4 * (Qc + Qr) <= 1e-5),
        "Axb_ok": np.all([np.all(Ai[0]*Qp + Ai[1]*Qc + Ai[2]*Qr - bi <= 1e-5) for Ai, bi in zip(A, b)])
    }

    return df_out, final_cost, success, message, constraints_satisfied, model, cost_history
