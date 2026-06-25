import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho RandomForestClassifier.
        data_path  : duong dan den file du lieu huan luyen (co the la nhieu file cach nhau dau phay).
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    # Doc du lieu huan luyen - ho tro nhieu file (noi cach nhau bang dau phay)
    data_files = [p.strip() for p in data_path.split(",") if os.path.exists(p.strip())]
    df_train = pd.concat([pd.read_csv(f) for f in data_files], ignore_index=True)
    df_eval  = pd.read_csv(eval_path)

    print(f"Du lieu huan luyen: {len(df_train)} mau tu {data_files}")

    # Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval  = df_eval.drop(columns=["target"])
    y_eval  = df_eval["target"]

    # Cau hinh MLflow tracking URI
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)

    with mlflow.start_run():

        # Ghi nhan cac sieu tham so
        mlflow.log_params(params)
        mlflow.log_param("n_train_samples", len(df_train))

        # Khoi tao va huan luyen RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Du doan tren tap danh gia va tinh chi so
        preds = model.predict(X_eval)
        acc   = accuracy_score(y_eval, preds)
        f1    = f1_score(y_eval, preds, average="weighted")

        # Ghi nhan chi so vao MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        # In ket qua ra man hinh
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Luu metrics ra file outputs/metrics.json
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        # Luu mo hinh ra file models/model.pkl
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    # Dung ca phase1 + phase2 khi co the de dat accuracy cao hon
    data_path = "data/train_phase1.csv,data/train_phase2.csv"
    train(params, data_path=data_path)
