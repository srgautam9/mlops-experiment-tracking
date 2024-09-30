import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

mlflow.set_experiment("testing_experiment")
# Load the dataset
df = pd.read_csv('data/bike_sharing.csv')
X = df[['temp', 'humidity', 'windspeed']]  # Example features
y = df['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_log_model(model, model_name):
    with mlflow.start_run():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        # Log parameters, metrics, and model
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, model_name)

        print(f"{model_name} MSE: {mse}")

# Train Linear Regression
lin_reg = LinearRegression()
train_and_log_model(lin_reg, "Linear_Regression")

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
train_and_log_model(rf, "Random_Forest")

