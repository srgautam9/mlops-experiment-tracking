import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

mlflow.set_experiment("modified_experiment")

df = pd.read_csv('data/bikeshare.csv')
X = df[['temp', 'humidity', 'windspeed']] 
y = df['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_log_model(model, model_name, hyperparams={}):
    with mlflow.start_run():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        mlflow.log_param("model_name", model_name)
        for param, value in hyperparams.items():
            mlflow.log_param(param, value)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, model_name)

        print(f"{model_name} MSE: {mse}")
        
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.title(f'{model_name} Predictions vs Actual')
        plt.xlabel('Actual')
        plt.ylabel('Predictions')
        plt.savefig(f'{model_name}_pred_vs_actual.png')
        plt.close()

        mlflow.log_artifact(f'{model_name}_pred_vs_actual.png')

lin_reg = LinearRegression()
train_and_log_model(lin_reg, "Linear_Regression")

rf = RandomForestRegressor(n_estimators=100, random_state=42)
train_and_log_model(rf, "Random_Forest", hyperparams={"n_estimators": 100})

rf_tuned = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
train_and_log_model(rf_tuned, "Random_Forest_Tuned", hyperparams={"n_estimators": 200, "max_depth": 10})

decision_tree = DecisionTreeRegressor(max_depth=5)
train_and_log_model(decision_tree, "Decision_Tree", hyperparams={"max_depth": 5})

