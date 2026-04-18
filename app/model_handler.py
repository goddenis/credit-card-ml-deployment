import pickle

import pandas as pd

class ModelHandler:
    def __init__(self, model_path: str, feature_names_path: str):
        self.model = self._load_pickle(model_path)
        self.feature_names = self._load_pickle(feature_names_path)

    @staticmethod
    def _load_pickle(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def preprocess_input(self, data: dict) -> pd.DataFrame:
        missing_features = [col for col in self.feature_names if col not in data]
        if missing_features:
            raise ValueError(f"Missing required fields: {missing_features}")

        row = {col: data[col] for col in self.feature_names}
        return pd.DataFrame([row])

    def predict(self, data: dict):
        features = self.preprocess_input(data)
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0][1]

        return {
            "prediction": int(prediction),
            "probability": float(probability)
        }
