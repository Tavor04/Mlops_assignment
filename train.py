import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_openml

# Load the Ames Housing dataset
data = fetch_openml(name="house_prices", as_frame=True)
X = data.data.select_dtypes(include=['number']).dropna(axis=1)
y = data.target


# Check for missing values and fill them if necessary
X.fillna(X.mean(), inplace=True)  # Fill missing values with the mean

X = X.astype(float)
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up MLflow tracking
mlflow.set_experiment("Ames Housing Model Comparison")

def train_and_log_model(model, model_name):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        # Log parameters, metrics, and model
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("mse", mse)

        input_example = X_train.iloc[0:1]
        mlflow.sklearn.log_model(model, model_name, input_example=input_example)

        print(f"{model_name} model MSE: {mse}")
        return mse

# Train and log Linear Regression
lr = LinearRegression()
mse_lr = train_and_log_model(lr, "Linear_Regression")

# Train and log Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
mse_rf = train_and_log_model(rf, "Random_Forest")
# Compare MSE and save the best model
if mse_lr < mse_rf:
    print("Linear Regression performed better. Saving the model.")
    best_model = lr
    model_name = "Best_Linear_Regression_Model"
else:
    print("Random Forest performed better. Saving the model.")
    best_model = rf
    model_name = "Best_Random_Forest_Model"

# Save the best-performing model in MLflow's Model Registry
mlflow.sklearn.log_model(best_model, model_name)
