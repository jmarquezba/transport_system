"""
Trabajo 3 — IRNA 2026
Sistema Inteligente Integrado para Empresa de Transporte
Universidad Nacional de Colombia

Flask app con tres módulos de Deep Learning:
  Módulo 1 — Predicción de demanda (LSTM)
  Módulo 2 — Clasificación de conducción distractiva (ResNet18)
  Módulo 3 — Recomendación de destinos (NCF)

TODOS los modelos fueron entrenados con datos reales de Kaggle.
"""

import os
import io
import pickle
import base64
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, render_template, request

# ═══════════════════════════════════════════════════════════════════════════════
#  ARQUITECTURAS (deben coincidir exactamente con las usadas en entrenamiento)
# ═══════════════════════════════════════════════════════════════════════════════

class LSTMDemanda(nn.Module):
    """LSTM mejorado: 3 capas, hidden=128, con cabeza MLP."""
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, dropout=0.25):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def build_resnet18(num_classes: int) -> nn.Module:
    """ResNet18 con la cabeza exacta usada durante el entrenamiento."""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


class NCF(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int = 32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),          nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),           nn.ReLU(),
            nn.Linear(32, 1),            nn.Sigmoid(),
        )

    def forward(self, user_ids, item_ids):
        return self.mlp(torch.cat([self.user_emb(user_ids),
                                   self.item_emb(item_ids)], dim=1)).squeeze()


# ═══════════════════════════════════════════════════════════════════════════════
#  CARGA DE MODELOS
# ═══════════════════════════════════════════════════════════════════════════════

MODELS_DIR = Path(__file__).parent / "models"
_device    = torch.device("cpu")


def _load_lstm():
    """Carga estado LSTM + scalers + metadata de rutas."""
    lstm_pt  = MODELS_DIR / "lstm_demanda.pt"
    sc_pkl   = MODELS_DIR / "scaler_demanda.pkl"
    meta_pkl = MODELS_DIR / "routes_metadata.pkl"

    if not all(p.exists() for p in [lstm_pt, sc_pkl, meta_pkl]):
        raise FileNotFoundError("Artefactos LSTM no encontrados. Ejecuta train_lstm.py primero.")

    states  = torch.load(lstm_pt, map_location=_device)
    with open(sc_pkl,   "rb") as f: scalers  = pickle.load(f)
    with open(meta_pkl, "rb") as f: metadata = pickle.load(f)
    return states, scalers, metadata


def _load_resnet18():
    """Carga ResNet18 — lee num_classes dinámicamente del .pt guardado."""
    pt_path  = MODELS_DIR / "resnet18_driver.pt"
    pkl_path = MODELS_DIR / "class_names.pkl"

    if not pt_path.exists():
        raise FileNotFoundError("resnet18_driver.pt no encontrado. Ejecuta train_cnn.py primero.")

    # El .pt puede ser un dict con state_dict + metadata, o solo state_dict
    data = torch.load(pt_path, map_location=_device)
    if isinstance(data, dict) and "state_dict" in data:
        state_dict  = data["state_dict"]
        num_classes = int(data.get("num_classes", 0))
        class_names = data.get("class_names", [])
    else:
        state_dict  = data
        num_classes = 0
        class_names = []

    # Leer class_names.pkl si existe
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            class_names = pickle.load(f)

    # Inferir num_classes del ultimo Linear del state_dict si no está disponible
    if num_classes == 0:
        for key in reversed(list(state_dict.keys())):
            if "weight" in key and len(state_dict[key].shape) == 2:
                num_classes = state_dict[key].shape[0]
                break

    if num_classes == 0:
        raise ValueError("No se pudo determinar el número de clases del modelo CNN.")

    # Rellenar class_names si faltan
    if len(class_names) != num_classes:
        class_names = [f"c{i}" for i in range(num_classes)]

    model = build_resnet18(num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model, class_names, num_classes


def _load_ncf():
    """Carga modelo NCF + metadata."""
    pt_path  = MODELS_DIR / "ncf_model.pt"
    pkl_path = MODELS_DIR / "ncf_metadata.pkl"

    if not all(p.exists() for p in [pt_path, pkl_path]):
        raise FileNotFoundError("Artefactos NCF no encontrados. Ejecuta train_ncf.py primero.")

    with open(pkl_path, "rb") as f:
        metadata = pickle.load(f)

    model = NCF(metadata["n_users"], metadata["n_items"])
    model.load_state_dict(torch.load(pt_path, map_location=_device))
    model.eval()
    return model, metadata


# ── Carga en startup ──────────────────────────────────────────────────────────
print("Cargando modelos...")

try:
    lstm_states, scaler_demanda, routes_metadata = _load_lstm()
    _lstm_routes = list(lstm_states.keys())
    print(f"  [OK] LSTM — rutas: {_lstm_routes}")
except Exception as e:
    lstm_states = scaler_demanda = routes_metadata = None
    _lstm_routes = []
    print(f"  [WARN] LSTM no cargado: {e}")

try:
    resnet_model, class_names_cnn, num_cnn_classes = _load_resnet18()
    print(f"  [OK] ResNet18 — {num_cnn_classes} clases: {class_names_cnn}")
except Exception as e:
    resnet_model = None
    class_names_cnn = []
    num_cnn_classes = 0
    print(f"  [WARN] ResNet18 no cargado: {e}")

try:
    ncf_model, ncf_metadata = _load_ncf()
    print(f"  [OK] NCF — {ncf_metadata['n_users']} usuarios, {ncf_metadata['n_items']} ítems")
except Exception as e:
    ncf_model = ncf_metadata = None
    print(f"  [WARN] NCF no cargado: {e}")

# ── Medidas preventivas por tipo de distracción ───────────────────────────────
PREVENTIVE_MEASURES = {
    "c0": "Conductor en estado seguro. Mantener buenas prácticas de manejo.",
    "c1": "⚠️ Texto (der): Prohibir uso de teléfono al conducir. Instalar bloqueador automático.",
    "c2": "⚠️ Llamada (der): Usar manos libres. Detener el vehículo para llamadas largas.",
    "c3": "⚠️ Texto (izq): Prohibir uso de teléfono. Política cero tolerancia.",
    "c4": "⚠️ Llamada (izq): Instalar sistema de llamadas por voz integrado al vehículo.",
    "c5": "⚠️ Controles: Capacitar conductores para operar controles sin desviar la vista.",
    "c6": "⚠️ Bebiendo: Prohibir consumo de alimentos/bebidas al conducir.",
    "c7": "⚠️ Alcanzando: Asegurar objetos antes de iniciar la marcha.",
    "c8": "⚠️ Arreglo personal: Política de tolerancia cero al arreglo personal en marcha.",
    "c9": "⚠️ Conversando: Limitar distracciones con pasajeros en maniobras críticas.",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  FLASK APP
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


# ── Páginas ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/modulo1")
def modulo1():
    return render_template("modulo1.html", destinations=_lstm_routes or ["Sin rutas cargadas"])

@app.route("/modulo2")
def modulo2():
    classes = [{"id": i, "name": n} for i, n in enumerate(class_names_cnn)]
    return render_template("modulo2.html", classes=classes)

@app.route("/modulo3")
def modulo3():
    sample_users = []
    if ncf_metadata:
        enc = ncf_metadata.get("user_encoder")
        if enc is not None:
            sample_users = list(enc.classes_[:10])
    return render_template("modulo3.html", sample_users=sample_users)


# ── API Módulo 1: Predicción de Demanda ───────────────────────────────────────

@app.route("/api/predict_demand", methods=["POST"])
def predict_demand():
    if lstm_states is None:
        return jsonify({"error": "Modelos LSTM no cargados."}), 503

    data        = request.get_json(silent=True) or {}
    destination = data.get("destination", _lstm_routes[0] if _lstm_routes else "")

    if destination not in lstm_states:
        return jsonify({"error": f"Destino '{destination}' no disponible. "
                                  f"Rutas válidas: {_lstm_routes}"}), 400

    try:
        LOOKBACK = routes_metadata[destination].get("lookback", 30)
        scaler   = scaler_demanda[destination]
        meta     = routes_metadata[destination]

        # Histórico: últimos 60 valores reales del conjunto de entrenamiento
        hist_vals  = meta.get("last_30d", [])
        hist_dates = meta.get("last_30d_dates", [])

        # Reconstruir modelo y hacer inferencia
        model = LSTMDemanda()
        model.load_state_dict(lstm_states[destination])
        model.eval()

        # Ventana para pronóstico: últimos LOOKBACK valores históricos escalados
        if hist_vals:
            raw_window = np.array(hist_vals[-LOOKBACK:], dtype=np.float32)
        else:
            raw_window = np.zeros(LOOKBACK, dtype=np.float32)

        window_scaled = scaler.transform(raw_window.reshape(-1, 1)).flatten().tolist()

        forecasts_scaled = []
        with torch.no_grad():
            for _ in range(30):
                x = torch.tensor(window_scaled[-LOOKBACK:], dtype=torch.float32)
                x = x.unsqueeze(0).unsqueeze(-1)
                p = float(model(x).item())
                p = max(0.0, min(1.0, p))
                forecasts_scaled.append(p)
                window_scaled.append(p)

        forecast_vals = np.clip(
            scaler.inverse_transform(
                np.array(forecasts_scaled).reshape(-1, 1)
            ).flatten(), 0, None
        ).tolist()

        # Fechas de pronóstico
        if hist_dates:
            last_dt = pd.Timestamp(hist_dates[-1])
        else:
            last_dt = pd.Timestamp.today()
        dates_fore = [(last_dt + timedelta(days=i + 1)).strftime("%Y-%m-%d")
                      for i in range(30)]

        return jsonify({
            "destination":  destination,
            "historical":   [round(v, 1) for v in hist_vals],
            "dates_hist":   hist_dates,
            "forecast":     [round(v, 1) for v in forecast_vals],
            "dates_fore":   dates_fore,
            "rmse":         round(meta["metrics"]["RMSE"],     2),
            "mae":          round(meta["metrics"]["MAE"],      2),
            "mape":         round(meta["metrics"]["MAPE (%)"], 2),
            "demand_mean":  round(meta.get("demand_mean", 0),  1),
        })
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# ── API Módulo 2: Clasificación de imagen ─────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
_img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


@app.route("/api/classify_image", methods=["POST"])
def classify_image():
    if resnet_model is None:
        return jsonify({"error": "Modelo CNN no cargado."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No se encontró el campo 'image' en la solicitud."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Nombre de archivo vacío."}), 400

    try:
        img    = Image.open(file.stream).convert("RGB")
        tensor = _img_transform(img).unsqueeze(0)

        with torch.no_grad():
            logits = resnet_model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze(0).numpy().tolist()

        pred_idx    = int(np.argmax(probs))
        pred_label  = class_names_cnn[pred_idx] if pred_idx < len(class_names_cnn) else f"c{pred_idx}"
        confidence  = float(probs[pred_idx])
        measure     = PREVENTIVE_MEASURES.get(pred_label,
                      PREVENTIVE_MEASURES.get(f"c{pred_idx}", "Revisar comportamiento del conductor."))

        return jsonify({
            "predicted_class":     pred_idx,
            "class_name":          pred_label,
            "confidence":          round(confidence, 4),
            "preventive_measure":  measure,
            "all_probabilities":   [round(p, 4) for p in probs],
            "class_names":         class_names_cnn,
        })
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# ── API Módulo 3: Recomendaciones ────────────────────────────────────────────

@app.route("/api/get_recommendations", methods=["POST"])
def get_recommendations():
    if ncf_model is None or ncf_metadata is None:
        return jsonify({"error": "Modelo NCF no cargado."}), 503

    data       = request.get_json(silent=True) or {}
    user_id_in = data.get("user_id")
    top_k      = int(data.get("top_k", 5))

    if user_id_in is None:
        return jsonify({"error": "Campo 'user_id' requerido."}), 400

    try:
        user_id = int(user_id_in)
    except (ValueError, TypeError):
        return jsonify({"error": "user_id debe ser entero."}), 400

    user_enc_obj = ncf_metadata["user_encoder"]
    item_enc_obj = ncf_metadata["item_encoder"]
    dest_catalog = ncf_metadata["dest_catalog"]

    if user_id not in user_enc_obj.classes_:
        samples = list(user_enc_obj.classes_[:8])
        return jsonify({
            "error": (f"UserID {user_id} no encontrado en el dataset de Kaggle. "
                      f"Prueba con: {samples}")
        }), 400

    try:
        user_enc = int(user_enc_obj.transform([user_id])[0])

        # Construir tensores para todos los ítems válidos
        valid_items, valid_encs = [], []
        for item in dest_catalog:
            did = item["DestinationID"]
            if did in item_enc_obj.classes_:
                valid_items.append(item)
                valid_encs.append(int(item_enc_obj.transform([did])[0]))

        u_tensor = torch.tensor([user_enc] * len(valid_encs), dtype=torch.long)
        i_tensor = torch.tensor(valid_encs,                   dtype=torch.long)

        with torch.no_grad():
            scores = ncf_model(u_tensor, i_tensor).numpy()
        scores = np.atleast_1d(scores)

        top_idx = np.argsort(scores)[::-1][:top_k]
        recs = []
        for rank, idx in enumerate(top_idx, 1):
            item = valid_items[idx]
            recs.append({
                "rank":        rank,
                "destination": item.get("Name",        "N/A"),
                "category":    item.get("Type",        item.get("category", "N/A")),
                "state":       item.get("State",       "N/A"),
                "popularity":  round(float(item.get("Popularity", 0)), 2),
                "best_time":   item.get("BestTimeToVisit", "N/A"),
                "score":       round(float(scores[idx]), 4),
                "description": item.get("description", ""),
            })

        return jsonify({
            "user_id":         user_id,
            "recommendations": recs,
            "metrics": {
                "Precision@5": round(ncf_metadata["metrics"].get("Precision@5", 0), 4),
                "Recall@5":    round(ncf_metadata["metrics"].get("Recall@5",    0), 4),
                "NDCG@10":     round(ncf_metadata["metrics"].get("NDCG@10",     0), 4),
            },
        })
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# ── Health-check ─────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    return jsonify({
        "lstm_loaded":    lstm_states    is not None,
        "cnn_loaded":     resnet_model   is not None,
        "ncf_loaded":     ncf_model      is not None,
        "lstm_routes":    _lstm_routes,
        "cnn_classes":    class_names_cnn,
        "ncf_users":      int(ncf_metadata["n_users"])  if ncf_metadata else 0,
        "ncf_items":      int(ncf_metadata["n_items"])  if ncf_metadata else 0,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") in ("1", "true", "True")
    app.run(host="0.0.0.0", port=port, debug=debug)
