import os
import pickle
import numpy as np
from pathlib import Path

class GridSightPredictor:
    def __init__(self):
        # Resolve models path relative to ml-service root
        self.model_path = Path(__file__).resolve().parents[1] / "models"
        self.congestion_model_file = os.getenv("CONGESTION_MODEL_FILE", "congestion_model.pkl")
        self.payload = None

    def load_model(self):
        model_full_path = self.model_path / self.congestion_model_file
        if model_full_path.exists():
            try:
                with open(model_full_path, "rb") as f:
                    self.payload = pickle.load(f)
                print(f"[ML Models] Congestion payload loaded successfully from {model_full_path}")
            except Exception as e:
                print(f"[ML Models] ERROR loading model: {e}")
        else:
            print(f"[ML Models] Congestion model not found at {model_full_path}. Fallback enabled.")

    def run_prediction(self, features: np.ndarray) -> float:
        # Backward compatibility for raw array inputs
        if self.payload:
            try:
                model = self.payload["model"]
                return float(model.predict(features)[0])
            except Exception as e:
                print(f"[ML Models] Prediction inference error: {e}")
        return float(np.sum(features) * 0.05)

    def predict_delay_impact(
        self, 
        latitude: float, 
        longitude: float, 
        hour: int, 
        vehicle_type: str, 
        violation_type: str, 
        cluster_size: int = 1,
        is_sensitive: int = 0
    ) -> float:
        if self.payload:
            try:
                model = self.payload["model"]
                v_map = self.payload["vehicle_map"]
                t_map = self.payload["violation_map"]

                # Map categories using model dictionaries
                v_code = v_map.get(str(vehicle_type).strip().upper(), 3)  # default: MOTOR CAR (3)
                t_code = t_map.get(str(violation_type).strip().upper(), 0)  # default: UNKNOWN (0)

                X_in = np.array([[
                    latitude,
                    longitude,
                    hour,
                    v_code,
                    t_code,
                    is_sensitive,
                    cluster_size
                ]])
                return float(model.predict(X_in)[0])
            except Exception as e:
                print(f"[ML Models] Category prediction failed: {e}")
                
        # Heuristic fallback calculation if model is missing
        vtype = str(vehicle_type).upper()
        vehicle_delay = 14.0 if "HEAVY" in vtype or "BUS" in vtype or "TRUCK" in vtype else (4.0 if "CAR" in vtype or "SUV" in vtype else 0.5)
        sensitive_delay = 8.0 if is_sensitive == 1 else 0.0
        return float(2.0 + vehicle_delay + sensitive_delay + min(15.0, cluster_size * 1.5))

# Singleton instance for global usage
predictor = GridSightPredictor()
predictor.load_model()
