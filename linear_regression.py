import argparse
import os
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import joblib


def model_fn(model_dir):
    """
    Load the trained model from the model_dir.
    This will be used by SageMaker during endpoint deployment.
    """
    model_path = os.path.join(model_dir, "best_model.joblib")
    model = joblib.load(model_path)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Default hyperparameters (can be overridden from SageMaker)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.0001)

    # Data & model directories (SageMaker will pass these automatically)
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "train"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST", "test"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))

    args = parser.parse_args()
    print("Training with arguments:", args)

    # ---------- Load Data ----------
    train_file = os.path.join(args.train, "train_data.csv")
    test_file = os.path.join(args.test, "test_data.csv")

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

    # ---------- Hyperparameter Grid ----------
    param_grid = {
        "alpha": [0.01],
        "eta0": [0.01, 0.3],  # learning rate
        "max_iter": [10],
    }

    # ---------- Define SGDRegressor ----------
    sgd = SGDRegressor(
        learning_rate="constant",
        random_state=42
    )

    # ---------- Grid Search ----------
    grid = GridSearchCV(
        estimator=sgd,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=3,
        verbose=1
    )

    grid.fit(X_train, y_train)

    # ---------- Best Hyperparameters ----------
    print("Best Hyperparameters:", grid.best_params_)
    best_model = grid.best_estimator_

    # ---------- Evaluate ----------
    preds = best_model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print("Test MSE: {:.4f}".format(mse))

    # ---------- Save Model ----------
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "best_model.joblib")
    joblib.dump(best_model, model_path)
    print("Best model saved at {}".format(model_path))
