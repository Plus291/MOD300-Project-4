import os 
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from PIL import Image
from sklearn.cluster import KMeans


try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow is not installed. LSTM functions will not be available.")



"""
#Topic 2 Task 0: Reproduce Ebola Plots
"""
def load_ebola_data(filename):
    assert isinstance(filename, str), "Filename must be a string"
    assert filename.endswith('.dat'), "Filename must end with .dat"
    filepath = os.path.join('data', filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    dates = []
    new_cases = []

    for line in lines[1:]:
        if line.strip() and not line.startswith('#'):
            parts =line.strip().split()
            if len(parts) >= 3:
                dates.append(parts[0])
                try:
                    new_cases.append(float(parts[2]))
                except ValueError:
                    new_cases.append(0.0)

    assert len(dates) > 0, "No data found in the file."

    first_date = datetime.strptime(dates[0], '%Y-%m-%d')
    days = []
    for date_str in dates:
        current_date = datetime.strptime(date_str, '%Y-%m-%d')
        days_since = (current_date - first_date).days
        days.append(days_since)

    days = np.array(days)
    new_cases = np.array(new_cases)
    cumulative_cases = np.cumsum(new_cases)

    assert len(days) == len(new_cases) == len(cumulative_cases), "Data arrays must be of the same length."

    return days, new_cases, cumulative_cases

def sezr_model(y, t, beta0, lam, sigma, gamma, N):
    assert len(y) == 4, "State vector y must have four elements: S, E, Z, R"
    assert all(val >= 0 for val in y), "State vector values must be non-negative"
    assert beta0 >= 0, "beta0 must be non-negative"
    assert lam >= 0, "lam must be non-negative"
    assert sigma > 0, "sigma must be positive"
    assert gamma > 0, "gamma must be positive"
    assert N > 0, "Population N must be positive"

    S, E, Z, _ = y
    beta_t = beta0 * np.exp(-lam * t)

    dS = -beta_t * S * Z / N
    dE = beta_t * S * Z / N - sigma * E
    dZ = sigma * E - gamma * Z
    dR = gamma * Z

    return [dS, dE, dZ, dR]

def solve_sezr(beta0, lam, t_max=700, N=1e7, sigma=1.0/9.7, gamma=1.0/7.0):
    assert beta0 >= 0 and lam >= 0, "beta0 and lam must be non-negative"
    assert N > 0 and sigma > 0 and gamma > 0, "Population N, sigma, and gamma must be positive"
    assert t_max > 0, "t_max must be positive"

    S0 = N - 1
    E0 = 1
    Z0 = 0
    R0 = 0

    t = np.linspace(0, t_max, t_max + 1)
    solution = odeint(sezr_model, [S0, E0, Z0, R0], t, args=(beta0, lam, sigma, gamma, N))

    return t, solution

def plot_ebola_data(days, new_cases, cumulative, country_name):
    assert len(days) == len(new_cases) == len(cumulative), "Data arrays must be of the same length"
    assert len(days) > 0, "Data arrays must not be empty"

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot new cases on left axis (red)
    ax1.scatter(days, new_cases, color='red', s=80,
                facecolors='none', edgecolors='red',
                linewidth=2, alpha=0.7, label='Number of outbreaks',
                zorder=5)
    ax1.set_xlabel('Days since first outbreak', fontsize=12)
    ax1.set_ylabel('Number of outbreaks', fontsize=12, color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.set_xlim([0, max(days) + 50])
    ax1.set_ylim([0, max(new_cases) * 1.15])
    ax1.grid(True, alpha=0.3)

    # Plot cumulative cases on right axis (black)
    ax2 = ax1.twinx()
    ax2.plot(days, cumulative, marker='s', markersize=4,
             linewidth=1.5, color='black',
             markerfacecolor='black', markeredgecolor='black',
             markeredgewidth=1, label='Cumulative number of outbreaks',
             zorder=4)
    ax2.set_ylabel('Cumulative number of outbreaks',
                   fontsize=12, color='black',
                   rotation=270, labelpad=20)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim([0, max(cumulative) * 1.15])
    plt.title(f'Ebola outbreaks in {country_name}',
              fontsize=14, fontweight='bold')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper left', fontsize=10)

    plt.tight_layout()

    return fig, (ax1, ax2)

def plot_model_comparison(days, new_cases, cumulative, t, model_solution,
                         beta0, lam, country_name, N=1e7):
    assert len(days) > 0, "Days array must not be empty"
    assert model_solution.shape[1] == 4, "Model solution must have four columns: S, E, Z, R"

    # modelt results
    S = model_solution[:, 0]
    model_cumulative = N - S
    model_new = np.diff(model_cumulative, prepend=0)

    #calculate r-squared
    model_cum_interp = np.interp(days, t, model_cumulative)
    ss_res = np.sum((cumulative - model_cum_interp) ** 2)
    ss_tot = np.sum((cumulative - np.mean(cumulative)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    #create plot
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # New cases (left axis)
    ax1.scatter(days, new_cases, c='red', s=80,
                facecolors='none', edgecolors='red', lw=2,
                label='Data: New cases', zorder=5)
    ax1.plot(t, model_new, 'b--', lw=2,
             label='Model: New cases')
    ax1.set_xlabel('Days since first outbreak', fontsize=12)
    ax1.set_ylabel('Number of outbreaks', fontsize=12, color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(alpha=0.3)

    # Cumulative cases (right axis)
    ax2 = ax1.twinx()
    ax2.plot(days, cumulative, 'ks-', ms=4, lw=1.5,
             label='Data: Cumulative', zorder=4)
    ax2.plot(t, model_cumulative, 'g--', lw=2,
             label='Model: Cumulative')
    ax2.set_ylabel('Cumulative number of outbreaks',
                   fontsize=12, color='black',
                   rotation=270, labelpad=20)
    ax2.tick_params(axis='y', labelcolor='black')
    plt.title(f'{country_name}: β₀={beta0:.4f}, λ={lam:.6f} (R²={r_squared:.3f})',
              fontsize=14, fontweight='bold')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc='upper left', fontsize=10)

    plt.tight_layout()

    return fig, (ax1, ax2), r_squared


"""
#Topic 2 Task 1: Train a line with linear regression on the data for ebola outbreaks in Guinea, Liberia, and Sierra Leone
"""
def train_linear_regression(days, cumulative):
    assert len(days) == len(cumulative), "Days and cumulative arrays must be of the same length"
    assert len(days) > 1, "At least two data points are required for linear regression"
    assert np.all(np.isfinite(days)), "Days array must contain finite values"
    assert np.all(np.isfinite(cumulative)), "Cumulative array must contain finite values"

    X = days.reshape(-1, 1)
    y = cumulative

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    r2 = r2_score(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)

    metrics = {
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'slope': model.coef_[0],
        'intercept': model.intercept_
    }

    return model, predictions, metrics

def plot_linear_regression(days, new_cases, cumulative, predictions, metrics, country_name):
    assert len(days) == len(new_cases) == len(cumulative) == len(predictions), "All data arrays must be of the same length"
    assert len(days) > 0, "Data arrays must not be empty"
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    #plot new cases on left axis (red)
    ax1.scatter(days, new_cases, color='red', s=80,
                facecolors='none', edgecolors='red',
                linewidth=2, alpha=0.7, label='Number of outbreaks',
                zorder=5)
    ax1.set_xlabel('Days since first outbreak', fontsize=12)
    ax1.set_ylabel('Number of outbreaks', fontsize=12, color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True, alpha=0.3)

    #plot cumulative cases on right axis (black)
    ax2 = ax1.twinx()
    ax2.plot(days, cumulative, 'ks-', ms=4, lw=1.5,
             label='Data: Cumulative', zorder=4)
    ax2.plot(days, predictions, 'b--', lw=2.5,
             label='Linear Regression', zorder=3)
    ax2.set_ylabel('Cumulative number of outbreaks',
                   fontsize=12, color='black',
                   rotation=270, labelpad=20)
    ax2.tick_params(axis='y', labelcolor='black')


    plt.title(f'{country_name} - Linear Regression\n'
              f'y = {metrics["slope"]:.2f}x + {metrics["intercept"]:.0f} '
              f'(R² = {metrics["r2"]:.3f}, RMSE = {metrics["rmse"]:.0f})',
              fontsize=13, fontweight='bold')
    
    #combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper left', fontsize=10)
    
    plt.tight_layout()

    return fig, (ax1, ax2)

def evaluate_linear_model(days, cumulative, model):
    assert len(days) == len(cumulative), "arrays must be of the same length"
    assert len(days) > 0, "arrays must not be empty"

    X = days.reshape(-1, 1)
    predictions = model.predict(X)

    r2 = r2_score(cumulative, predictions)
    mse = mean_squared_error(cumulative, predictions)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(cumulative - predictions))

    return {
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'slope': model.coef_[0],
        'intercept': model.intercept_
    }

"""
# Topic 2 Task 2: Train a better fitting function than a single line with linear regression on the data for the
three countries
"""
def train_polynomial_regression(days, cumulative, degree=2):
    assert len(days) ==len(cumulative), "Days and cumulative arrays must be of the same length"
    assert len(days) > degree, "Number of data points must be greater than polynomial degree"
    assert degree >= 1, "Degree must be at least 1"
    assert degree <= 10, "Degree must be 10 or less to avoid overfitting"
    assert np.all(np.isfinite(days)), "Days array must contain finite values"
    assert np.all(np.isfinite(cumulative)), "Cumulative array must contain finite values"

    X = days.reshape(-1, 1)
    y = cumulative

    model = Pipeline([
        ('poly_features', PolynomialFeatures(degree=degree)),
        ('linear_regression', LinearRegression())
    ])

    model.fit(X, y)
    predictions = model.predict(X)

    r2 = r2_score(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)

    coefficients = model.named_steps['linear_regression'].coef_

    metrics = {
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'degree': degree,
    }
    return model, predictions, metrics, coefficients

def plot_polynomial_regression(days, new_cases, cumulative, predictions, metrics, country_name, degree):
    assert len(days) == len(new_cases) == len(cumulative) == len(predictions), \
        "All arrays must have same length"
    assert len(days) > 0, "Data arrays cannot be empty"

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot new cases on left axis (red)
    ax1.scatter(days, new_cases, color='red', s=80,
                facecolors='none', edgecolors='red', lw=2,
                label='Data: New cases', zorder=5, alpha=0.7)
    ax1.set_xlabel('Days since first outbreak', fontsize=12)
    ax1.set_ylabel('Number of outbreaks', fontsize=12, color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(alpha=0.3)

    # Plot cumulative cases on right axis
    ax2 = ax1.twinx()
    ax2.plot(days, cumulative, 'ks-', ms=4, lw=1.5,
             label='Data: Cumulative', zorder=4)
    ax2.plot(days, predictions, 'g--', lw=2.5,
             label=f'Polynomial (degree {metrics["degree"]})', zorder=3)
    ax2.set_ylabel('Cumulative number of outbreaks',
                   fontsize=12, color='black',
                   rotation=270, labelpad=20)
    ax2.tick_params(axis='y', labelcolor='black')

    # Title with metrics
    plt.title(f'{country_name} - Polynomial Regression (degree {metrics["degree"]})\n'
              f'R² = {metrics["r2"]:.3f}, RMSE = {metrics["rmse"]:.0f}',
              fontsize=13, fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper left', fontsize=10)

    plt.tight_layout()

    return fig, (ax1, ax2)

def compare_polynomial_degrees(days, cumulative, degrees=(2,3,4,5)):
    assert len(days) == len(cumulative), "Arrays must have same length"
    assert len(days) > max(degrees), \
        f"Need at least {max(degrees)+1} points for max degree {max(degrees)}"
    assert all(d >= 1 for d in degrees), "All degrees must be >= 1"

    results = {}
    best_r2 = -np.inf
    best_degree = degrees[0]
    
    for degree in degrees:
        model, predictions, metrics, coeffs = train_polynomial_regression(
            days, cumulative, degree
        )
    
        results[degree] = {
            'model': model,
            'predictions': predictions,
            'metrics': metrics,
            'coefficients': coeffs
        }

        if metrics['r2'] > best_r2:
            best_r2 = metrics['r2']
            best_degree = degree

    return results, best_degree

def plot_polynomial_comparison(days, cumulative,results, country_name):
    assert len(days) == len(cumulative), "Arrays must have same length"
    assert len(results) > 0, "Results dictionary cannot be empty"

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot real data
    ax.plot(days, cumulative, 'ko-', ms=4, lw=1.5,
            label='Real Data', zorder=10)

    # Plot each polynomial fit
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    for idx, (degree, result) in enumerate(sorted(results.items())):
        color = colors[idx % len(colors)]
        r2 = result['metrics']['r2']
        ax.plot(days, result['predictions'], '--', lw=2,
                color=color, label=f'Degree {degree} (R²={r2:.3f})',
                zorder=5-idx)
    ax.set_xlabel('Days since first outbreak', fontsize=12)
    ax.set_ylabel('Cumulative cases', fontsize=12)
    ax.set_title(f'{country_name} - Polynomial Degree Comparison',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    return fig, ax

"""
# Topic 2 Task 3:Train a NN network and predict the epidemic evolution
"""
def temporal_train_test_split(days, cumulative, train_ratio=0.7):
    assert len(days) == len(cumulative), "arrays must have same length"
    assert 0 < train_ratio < 1, "train_ratio must be between 0 and 1"
    assert len(days) > 10, "arrays must contain more than one element"

    split_index = int(len(days) * train_ratio)

    X_train = days[:split_index].reshape(-1, 1)
    X_test = days[split_index:].reshape(-1, 1)
    y_train = cumulative[:split_index]
    y_test = cumulative[split_index:]

    assert len(X_train) > 0 and len(X_test) > 0, "Train and test sets must not be empty"

    return X_train, X_test, y_train, y_test, split_index

def train_NN_network(X_train, y_train,X_test, y_test, hidden_layers=(100,50), max_iter=1000, random_state=42):
    assert len(X_train) > 0 and len(X_test) > 0, "Both sets must have data"
    assert len(X_train) == len(y_train), "X_train and y_train must match"
    assert len(X_test) == len(y_test), "X_test and y_test must match"
    assert len(hidden_layers) > 0, "Must have at least one hidden layer"

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    X_test_scaled = scaler_X.transform(X_test)

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        verbose=False
        )
    
    model.fit(X_train_scaled, y_train_scaled)

    train_pred_scaled = model.predict(X_train_scaled)
    test_pred_scaled = model.predict(X_test_scaled)
    
    train_predictions = scaler_y.inverse_transform(
        train_pred_scaled.reshape(-1, 1)
    ).ravel()
    test_predictions = scaler_y.inverse_transform(
        test_pred_scaled.reshape(-1, 1)
    ).ravel()

    train_r2 = r2_score(y_train, train_predictions)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    train_mae = mean_absolute_error(y_train, train_predictions)

    test_r2 = r2_score(y_test, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    test_mae = mean_absolute_error(y_test, test_predictions)
    metrics = {
        'train_r2': train_r2,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'hidden_layers': hidden_layers,
        'n_iter': model.n_iter_
    }
    return model, scaler_X, scaler_y, train_predictions, test_predictions, metrics

def plot_NN_network_results(days, cumulative, split_idx,train_predictions, test_predictions, metrics, country_name):
    assert len(days) == len(cumulative), "Arrays must have same length"
    assert split_idx > 0 and split_idx < len(days), "Invalid split index"

    fig, ax = plt.subplots(figsize=(14, 7))

    days_train = days[:split_idx]
    days_test = days[split_idx:]

    ax.plot(days, cumulative, 'ko-', ms=4, lw=1.5,
            label='Real Data', zorder=10)

    # Plot train predictions
    ax.plot(days_train, train_predictions, 'b--', lw=2.5,
            label=f'Train Predictions (R²={metrics["train_r2"]:.3f})',
            zorder=5)

    # Plot test predictions
    ax.plot(days_test, test_predictions, 'r--', lw=2.5,
            label=f'Test Predictions (R²={metrics["test_r2"]:.3f})',
            zorder=5)
    # Add vertical line at split point
    ax.axvline(x=days[split_idx], color='gray', linestyle=':',
               lw=2, label=f'Train/Test Split (day {days[split_idx]})')

    # Shade train/test regions
    ax.axvspan(days[0], days[split_idx], alpha=0.1, color='blue',
               label='Training Period')
    ax.axvspan(days[split_idx], days[-1], alpha=0.1, color='red',
               label='Test Period')

    ax.set_xlabel('Days since first outbreak', fontsize=12)
    ax.set_ylabel('Cumulative cases', fontsize=12)
    ax.set_title(f'{country_name} - Neural Network Predictions\n'
                 f'Architecture: {metrics["hidden_layers"]}, '
                 f'Train R²={metrics["train_r2"]:.3f}, '
                 f'Test R²={metrics["test_r2"]:.3f}',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig, ax

def compare_NN_architectures(X_train, y_train, X_test, y_test, architectures=None):
    if architectures is None:
        architectures = [(50,), (100,), (100, 50), (100, 100), (150, 100, 50)]

    results = {}
    best_r2 = -np.inf
    best_architecture = architectures[0]

    for arch in architectures:
        model, scaler_X, scaler_y, train_pred, test_pred, metrics = train_NN_network(
            X_train, y_train, X_test, y_test, hidden_layers=arch, max_iter=1000
        )

        results[arch] = {
            'model': model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'train_predictions': train_pred,
            'test_predictions': test_pred,
            'metrics': metrics
        }

        if metrics['test_r2'] > best_r2:
            best_r2 = metrics['test_r2']
            best_architecture = arch

    return results, best_architecture
    
"""
 # Topic 2 Task 4: Train a LSTM (a NN specialized for time series) and predict the epidemic evolution
"""   
def create_sequences(data, lookback=10):
    assert len(data) > lookback, "Data length must be greater than lookback period"
    assert lookback > 1, "Lookback period must be at least 1"
    
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])
    
    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def temporal_train_test_split_lstm(days, cumulative, lookback=10, train_ratio=0.7):
    assert len(days) == len(cumulative), "arrays must have same length"
    assert 0 < train_ratio < 1, "train_ratio must be between 0 and 1"
    assert len(days) > lookback + 10, "arrays must contain more than lookback + 1 elements"

    scaler = StandardScaler()
    cumulative_scaled = scaler.fit_transform(cumulative.reshape(-1, 1)).ravel()

    train_size = int(len(cumulative_scaled) * train_ratio)
    train_data = cumulative_scaled[:train_size]
    test_data = cumulative_scaled[train_size - lookback:]

    X_train, y_train = create_sequences(train_data, lookback)
    X_test, y_test = create_sequences(test_data, lookback)

    return X_train, y_train, X_test, y_test, scaler, train_size

def build_lstm_model(lookback, units=50, dropout=0.2, layers=2):
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow/Keras not available. Install with: pip install tensorflow")
    
    assert lookback > 0, "Lookback must be positive"
    assert units > 0, "Units must be positive"
    assert 0 <= dropout < 1, "Dropout must be between 0 and 1"
    assert layers >= 1, "Must have at least 1 LSTM layer"

    model = Sequential()
    model.add(Input(shape=(lookback, 1)))
    if layers == 1:
        model.add(LSTM(units))
    else:
        model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(dropout))
        for _ in range(1, layers - 1):
            model.add(LSTM(units, return_sequences=True))
            model.add(Dropout(dropout))

        model.add(LSTM(units))
    
    model.add(Dropout(dropout))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_lstm_model(X_train, y_train, X_test, y_test,
                     lookback=10, units=50, dropout=0.2,layers=2, 
                     epochs=100, batch_size=32,verbose=0):
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow/Keras not available")
    
    model = build_lstm_model(lookback, units, dropout, layers)

    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose
    )

    train_predictions = model.predict(X_train, verbose=0).flatten()
    test_predictions = model.predict(X_test, verbose=0).flatten()

    train_r2 = r2_score(y_train, train_predictions)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    train_mae = mean_absolute_error(y_train, train_predictions)

    test_r2 = r2_score(y_test, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    test_mae = mean_absolute_error(y_test, test_predictions)

    metrics = {
        'train_r2': train_r2,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'lookback': lookback,
        'units': units,
        'layers': layers,
        'epochs_trained': len(history.history['loss'])
    }

    return model, history, train_predictions, test_predictions, metrics
    
def plot_lstm_results(days, cumulative, lookback, train_size, scaler,
                     train_predictions, test_predictions, metrics, country_name):
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow/Keras not available")
    
    # Inverse transform predictions
    train_pred_original = scaler.inverse_transform(
        train_predictions.reshape(-1, 1)
    ).flatten()
    test_pred_original = scaler.inverse_transform(
        test_predictions.reshape(-1, 1)
    ).flatten()
    
    # Reconstruct timeline (accounting for lookback offset)
    train_days = days[lookback:train_size]
    test_days = days[train_size:train_size + len(test_pred_original)]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot real data
    ax.plot(days, cumulative, 'ko-', ms=4, lw=1.5,
            label='Real Data', zorder=10)
    
    # Plot LSTM predictions
    ax.plot(train_days, train_pred_original, 'b--', lw=2.5,
            label=f'LSTM Train (R²={metrics["train_r2"]:.3f})', zorder=5)
    ax.plot(test_days, test_pred_original, 'r--', lw=2.5,
            label=f'LSTM Test (R²={metrics["test_r2"]:.3f})', zorder=5)
    
    # Mark train/test split
    if train_size < len(days):
        ax.axvline(x=days[train_size], color='gray', linestyle=':',
                   lw=2, label=f'Train/Test Split (day {days[train_size]})')
    
    # Shade regions
    ax.axvspan(days[0], days[train_size], alpha=0.1, color='blue',
               label='Training Period')
    ax.axvspan(days[train_size], days[-1], alpha=0.1, color='red',
               label='Test Period')
    
    ax.set_xlabel('Days since first outbreak', fontsize=12)
    ax.set_ylabel('Cumulative cases', fontsize=12)
    ax.set_title(f'{country_name} - LSTM Predictions\n'
                 f'Lookback={metrics["lookback"]}, Units={metrics["units"]}, '
                 f'Layers={metrics["layers"]}, Train R²={metrics["train_r2"]:.3f}, '
                 f'Test R²={metrics["test_r2"]:.3f}',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ax

def compare_lstm_configurations(days, cumulative, configurations=None,
                                train_ratio=0.7, epochs=100, verbose=0):
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow/Keras not available")
    
    if configurations is None:
        configurations = [
            {'lookback': 7, 'units': 50, 'layers': 1, 'dropout': 0.2},
            {'lookback': 14, 'units': 50, 'layers': 2, 'dropout': 0.2},
            {'lookback': 21, 'units': 100, 'layers': 2, 'dropout': 0.2},
        ]

    results = {}
    best_test_r2 = -np.inf
    best_config = configurations[0]

    for config in configurations:
        lookback = config['lookback']
        X_train, y_train, X_test, y_test, scaler, train_size = temporal_train_test_split_lstm(
            days, cumulative, lookback=lookback, train_ratio=train_ratio)
        

        model, history, train_pred, test_pred, metrics = train_lstm_model(
            X_train, y_train, X_test, y_test,
            lookback=config['lookback'],
            units=config['units'],
            dropout=config['dropout'],
            layers=config['layers'],
            epochs=epochs,
            verbose=verbose
        )

        config_key = f"LB{config['lookback']}_U{config['units']}_L{config['layers']}"
        results[config_key] = {
            'model': model,
            'history': history,
            'train_predictions': train_pred,
            'test_predictions': test_pred,
            'metrics': metrics,
            'scaler': scaler,
            'train_size': train_size,
            'config': config
        }
        
        if metrics['test_r2'] > best_test_r2:
            best_test_r2 = metrics['test_r2']
            best_config = config_key
    
    return results, best_config

        
    
"""
# Test functions
"""
def test_load_ebola_data():
    """Test function for load_ebola_data."""
    # This would need actual test data
    print("Test: load_ebola_data - needs actual data file to test")
    return True

def test_sezr_model():
    """Test function for sezr_model."""
    y = [9999, 1, 0, 0]
    t = 0
    result = sezr_model(y, t, beta0=0.5, lam=0.001,
                       sigma=1/9.7, gamma=1/7.0, N=10000)
    assert len(result) == 4, "sezr_model must return 4 values"
    assert all(isinstance(x, (int, float)) for x in result), "Results must be numeric"
    print("✓ Test: sezr_model passed")
    return True

def test_solve_sezr():
    """Test function for solve_sezr."""
    t, sol = solve_sezr(beta0=0.5, lam=0.001, t_max=100, N=10000)
    assert len(t) == 101, "Time array should have t_max+1 points"
    assert sol.shape == (101, 4), "Solution should have shape (t_max+1, 4)"
    assert np.all(sol >= 0), "All compartments must be non-negative"
    print("✓ Test: solve_sezr passed")
    return True

def test_train_linear_regression():
    """ Test function for train_linear_regressin"""
    days = np.array([0, 10, 20, 30, 40, 50])
    cumulative = np.array([10, 50, 90, 130, 170, 210])

    model, predictions, metrics = train_linear_regression(days, cumulative)

    assert model is not None, "Model should not be None"
    assert len(predictions) == len(days), "Predictions should match input length"
    assert 'r2' in metrics, "Metrics should contain R²"
    assert 'slope' in metrics, "Metrics should contain slope"
    assert metrics['r2'] >= 0, "R² should be non-negative"
    print("✓ Test: train_linear_regression passed")
    return True

def test_train_polynomial_regression():
    """Test function for train_polynomial_regression."""
    # Create test data with quadratic pattern
    days = np.array([0, 10, 20, 30, 40, 50, 60])
    cumulative = np.array([10, 45, 110, 205, 330, 485, 670])  # Roughly quadratic

    # Test degree 2 (quadratic)
    model, predictions, metrics, coeffs = train_polynomial_regression(
        days, cumulative, degree=2
    )
    
    assert model is not None, "Model should not be None"
    assert len(predictions) == len(days), "Predictions should match input length"
    assert 'r2' in metrics, "Metrics should contain R²"
    assert 'degree' in metrics, "Metrics should contain degree"
    assert metrics['degree'] == 2, "Degree should be 2"
    assert metrics['r2'] >= 0, "R² should be non-negative"
    assert len(coeffs) > 0, "Coefficients should not be empty"
    print("✓ Test: train_polynomial_regression passed")
    return True

def test_temporal_train_test_split():
    """Test function for temporal_train_test_split."""
    days = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    cumulative = np.array([10, 30, 60, 100, 150, 210, 280, 360, 450, 550, 660])

    X_train, X_test, y_train, y_test, split_idx = temporal_train_test_split(
        days, cumulative, train_ratio=0.7
    )
    assert len(X_train) + len(X_test) == len(days), "Split should preserve total length"
    assert len(X_train) == len(y_train), "X_train and y_train must match"
    assert len(X_test) == len(y_test), "X_test and y_test must match"
    assert 0 < split_idx < len(days), "Split index in valid range"
    assert X_train[-1][0] < X_test[0][0], "Train data must come before test data"
    print("✓ Test: temporal_train_test_split passed")
    return True

def test_create_sequences():
    """Test function for create_sequences (LSTM)."""
    if not TENSORFLOW_AVAILABLE:
        print("⊗ Test: create_sequences skipped (TensorFlow not available)")
        return True
    
    # Create simple sequence
    data = np.array([10, 20, 30, 40, 50, 60])
    lookback = 2
    
    X, y = create_sequences(data, lookback)
    
    # Should create 4 sequences:
    # [10, 20] -> 30
    # [20, 30] -> 40
    # [30, 40] -> 50
    # [40, 50] -> 60
    assert X.shape == (4, 2, 1), f"X shape should be (4, 2, 1), got {X.shape}"
    assert y.shape == (4,), f"y shape should be (4,), got {y.shape}"
    assert y[0] == 30, "First target should be 30"
    assert y[-1] == 60, "Last target should be 60"
    print("✓ Test: create_sequences passed")
    return True


if __name__ == "__main__":
    print("Running tests...")
    test_sezr_model()
    test_solve_sezr()
    test_train_linear_regression()
    test_train_polynomial_regression()
    test_temporal_train_test_split()
    test_create_sequences()
    print("All tests completed!")

# Topic 1 Task 3:
def image_to_rgb_array(image_path):
    """
    A function to load an image file and convert it to RGB np.array
    
    Input
    -----
    image_path: str
        Path to the image file
        
    Output
    ------
    np.array(h, w, 3): A 3d map of each pixel in RGB encoding
    """
    img = Image.open(image_path)
    rgb_array = np.array(img)
    
    # If image has alpha channel (RGBA), take only RGB
    if len(rgb_array.shape) == 3 and rgb_array.shape[2] == 4:
        rgb_array = rgb_array[:, :, :3]
    
    return rgb_array

# Topic 1 Task 4:

def categorize_pixels(rgb_array):
    """
    Kategoriser piksler som rød, grønn eller blå
    """
    import numpy as np
    h, w, _ = rgb_array.shape
    categories = np.zeros((h, w), dtype=int)
    for i in range(h):
        for j in range(w):
            r, g, b = rgb_array[i, j]
            if r > g and r > b:
                categories[i, j] = 0
            elif g > r and g > b:
                categories[i, j] = 1
            elif b > r and b > g:
                categories[i, j] = 2
    return categories

# Topic 1 Task 5:

def apply_kmeans_clustering(rgb_array, n_clusters=6, visualize=True):
    """
    Apply K-means clustering to an RGB array
    
    Input
    -----
    rgb_array: np.array
        RGB image array (h, w, 3)
    n_clusters: int
        Number of clusters (default: 6)
    visualize: bool
        Whether to show the clustering visualization
        
    Output
    ------
    predicted_image: np.array
        Cluster labels reshaped to image dimensions (h, w)
    kmeans: KMeans object
        Fitted KMeans model
    """
    from sklearn.cluster import KMeans
    import numpy as np
    import matplotlib.pyplot as plt
    
    h, w, _ = rgb_array.shape
    rgb_flat = rgb_array.reshape(-1, 3)
    rgb_flat = rgb_flat.astype(np.float64)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(rgb_flat)
    cluster_labels = kmeans.labels_
    predicted_image = cluster_labels.reshape(h, w)
    
    print(f"Cluster centers:\n{kmeans.cluster_centers_}")
    print(f"Unique clusters: {np.unique(cluster_labels)}")
    print(f"Cluster distribution: {np.bincount(cluster_labels)}")
    
    if visualize:
        plt.figure(figsize=(10, 10))
        plt.imshow(predicted_image, cmap='viridis')
        plt.title(f'K-Means Clustering ({n_clusters} kluster)')
        plt.colorbar(label='Cluster ID')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return predicted_image, kmeans