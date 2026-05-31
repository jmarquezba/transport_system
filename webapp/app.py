"""
Trabajo 3 - IRNA: Neural Networks Web Application
Universidad Nacional de Colombia - 2026

Main Flask application with simulated ML model endpoints.
"""

import os
import json
import random
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ─────────────────────────────────────────────
#  LOAD REAL MODEL
# ─────────────────────────────────────────────
# Initialize ResNet18 structure
real_model = models.resnet18(pretrained=False)
in_features = real_model.fc.in_features
real_model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(in_features, 256),
    nn.ReLU(inplace=True),
    nn.BatchNorm1d(256),
    nn.Dropout(p=0.3),
    nn.Linear(256, 5) # 5 classes in the real dataset
)

# Load weights (check local folder first for deployment, then fallback to parent folder)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "resnet18_driver.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "modulo2_clasificacion", "models", "resnet18_driver.pt")

try:
    real_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    real_model.eval()
    print("Modelo ResNet18 real cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo ResNet18 real: {e}")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# ─────────────────────────────────────────────
#  CATALOGS
# ─────────────────────────────────────────────

DESTINATIONS_CATALOG = [
    {"id": 0,  "destination": "Cartagena de Indias",  "category": "playa",     "country": "Colombia",   "cost_usd": 850,  "description": "Ciudad amurallada con playas del Caribe y arquitectura colonial"},
    {"id": 1,  "destination": "Bogotá",                "category": "ciudad",    "country": "Colombia",   "cost_usd": 600,  "description": "Capital cultural con museos de clase mundial y gastronomía diversa"},
    {"id": 2,  "destination": "Medellín",              "category": "ciudad",    "country": "Colombia",   "cost_usd": 550,  "description": "Ciudad de la eterna primavera con innovación urbana y cultura"},
    {"id": 3,  "destination": "Santa Marta",           "category": "playa",     "country": "Colombia",   "cost_usd": 700,  "description": "Paraíso natural con playas cristalinas y sierra nevada cercana"},
    {"id": 4,  "destination": "San Andrés",            "category": "playa",     "country": "Colombia",   "cost_usd": 950,  "description": "Isla tropical con el mar de los siete colores y arrecifes de coral"},
    {"id": 5,  "destination": "Cusco",                 "category": "historia",  "country": "Perú",       "cost_usd": 1100, "description": "Capital del Imperio Inca con Machu Picchu y templos ancestrales"},
    {"id": 6,  "destination": "Lima",                  "category": "ciudad",    "country": "Perú",       "cost_usd": 750,  "description": "Capital gastronómica de América Latina con historia virreinal"},
    {"id": 7,  "destination": "Río de Janeiro",        "category": "playa",     "country": "Brasil",     "cost_usd": 1200, "description": "Ciudad maravillosa con playas icónicas, samba y carnaval"},
    {"id": 8,  "destination": "Buenos Aires",          "category": "ciudad",    "country": "Argentina",  "cost_usd": 900,  "description": "París de Sudamérica con tango, cultura y arquitectura europea"},
    {"id": 9,  "destination": "Ciudad de México",      "category": "ciudad",    "country": "México",     "cost_usd": 800,  "description": "Megalópolis con historia azteca, arte muralista y cocina prehispánica"},
    {"id": 10, "destination": "Cancún",                "category": "playa",     "country": "México",     "cost_usd": 1050, "description": "Resort de clase mundial con playas turquesa y ruinas mayas"},
    {"id": 11, "destination": "Patagonia Argentina",   "category": "naturaleza","country": "Argentina",  "cost_usd": 1400, "description": "Glaciares imponentes, Lagos y paisajes de fin del mundo"},
    {"id": 12, "destination": "Galápagos",             "category": "naturaleza","country": "Ecuador",    "cost_usd": 2200, "description": "Archipiélago único con fauna endémica que inspiró a Darwin"},
    {"id": 13, "destination": "Uyuni",                 "category": "naturaleza","country": "Bolivia",    "cost_usd": 650,  "description": "El salar más grande del mundo con reflejos infinitos de cielo"},
    {"id": 14, "destination": "Punta Cana",            "category": "playa",     "country": "Rep. Dom.",  "cost_usd": 1150, "description": "Paraíso caribeño con resorts todo incluido y playas blancas"},
    {"id": 15, "destination": "Havana",                "category": "historia",  "country": "Cuba",       "cost_usd": 750,  "description": "Ciudad atemporal con autos clásicos, son cubano y arquitectura barroca"},
    {"id": 16, "destination": "Santiago de Chile",     "category": "ciudad",    "country": "Chile",      "cost_usd": 850,  "description": "Moderna capital andina con acceso a viñedos y ski en los Andes"},
    {"id": 17, "destination": "Montevideo",            "category": "ciudad",    "country": "Uruguay",    "cost_usd": 700,  "description": "Ciudad costera tranquila con alta calidad de vida y cultura del mate"},
    {"id": 18, "destination": "San José",              "category": "naturaleza","country": "Costa Rica", "cost_usd": 950,  "description": "Puerta al ecoturismo con selvas, volcanes y biodiversidad única"},
    {"id": 19, "destination": "Panamá City",           "category": "ciudad",    "country": "Panamá",     "cost_usd": 800,  "description": "Hub financiero con el Canal, biodiversidad y vida nocturna"},
    {"id": 20, "destination": "Quito",                 "category": "historia",  "country": "Ecuador",    "cost_usd": 650,  "description": "Capital histórica en el centro del mundo con centro colonial UNESCO"},
    {"id": 21, "destination": "Asunción",              "category": "ciudad",    "country": "Paraguay",   "cost_usd": 500,  "description": "Ciudad cálida con historia guaraní y artesanía tradicional"},
    {"id": 22, "destination": "Trinidad y Tobago",     "category": "playa",     "country": "Trinidad",   "cost_usd": 1050, "description": "Doble isla caribeña con carnaval vibrante y aves exóticas"},
    {"id": 23, "destination": "Aruba",                 "category": "playa",     "country": "Aruba",      "cost_usd": 1350, "description": "Pequeña isla fuera de la zona de huracanes con playas perfectas"},
    {"id": 24, "destination": "Florianópolis",         "category": "playa",     "country": "Brasil",     "cost_usd": 900,  "description": "Isla mágica del sur de Brasil con playas de ensueño y surf"},
    {"id": 25, "destination": "Bariloche",             "category": "naturaleza","country": "Argentina",  "cost_usd": 1100, "description": "Suiza argentina con lagos glaciares, chocolate y ski de clase mundial"},
    {"id": 26, "destination": "Punta del Este",        "category": "playa",     "country": "Uruguay",    "cost_usd": 1200, "description": "Destino de lujo con playas exclusivas y vida nocturna sofisticada"},
    {"id": 27, "destination": "Antigua Guatemala",     "category": "historia",  "country": "Guatemala",  "cost_usd": 700,  "description": "Ciudad colonial perfectamente conservada entre volcanes activos"},
    {"id": 28, "destination": "Roatán",                "category": "playa",     "country": "Honduras",   "cost_usd": 850,  "description": "Isla caribeña con el segundo arrecife de coral más grande del mundo"},
    {"id": 29, "destination": "San Pedro de Atacama",  "category": "naturaleza","country": "Chile",      "cost_usd": 1000, "description": "Oasis en el desierto más árido con géiseres y lagunas altiplánicas"},
    {"id": 30, "destination": "Manaus",                "category": "naturaleza","country": "Brasil",     "cost_usd": 950,  "description": "Puerta de entrada al Amazonas con Teatro Ópera en la selva"},
    {"id": 31, "destination": "Iguazú",                "category": "naturaleza","country": "Argentina",  "cost_usd": 1100, "description": "Las cataratas más anchas del mundo en la triple frontera"},
    {"id": 32, "destination": "Valparaíso",            "category": "ciudad",    "country": "Chile",      "cost_usd": 650,  "description": "Ciudad bohemia y colorida declarada patrimonio UNESCO"},
    {"id": 33, "destination": "Bogotá (Zona Rosa)",    "category": "ciudad",    "country": "Colombia",   "cost_usd": 600,  "description": "El corazón gastronómico y de vida nocturna de la capital colombiana"},
    {"id": 34, "destination": "Tayrona",               "category": "naturaleza","country": "Colombia",   "cost_usd": 450,  "description": "Parque nacional con playas vírgenes y selva tropical milenaria"},
    {"id": 35, "destination": "Cali",                  "category": "ciudad",    "country": "Colombia",   "cost_usd": 500,  "description": "Capital mundial de la salsa con gente cálida y clima primaveral"},
    {"id": 36, "destination": "Villa de Leyva",        "category": "historia",  "country": "Colombia",   "cost_usd": 350,  "description": "Pueblo colonial de piedra con la plaza mayor más grande de Colombia"},
    {"id": 37, "destination": "Leticia",               "category": "naturaleza","country": "Colombia",   "cost_usd": 600,  "description": "Puerta colombiana al Amazonas con turismo indígena y biodiversidad"},
    {"id": 38, "destination": "Providencia",           "category": "playa",     "country": "Colombia",   "cost_usd": 800,  "description": "Isla declarada reserva de biósfera con el agua más clara del Caribe"},
    {"id": 39, "destination": "Nuquí",                 "category": "naturaleza","country": "Colombia",   "cost_usd": 700,  "description": "Paraíso del Pacífico colombiano con ballenas jorobadas y selva"},
    {"id": 40, "destination": "Popayán",               "category": "historia",  "country": "Colombia",   "cost_usd": 400,  "description": "Ciudad blanca colonial con Semana Santa patrimonio UNESCO y gastronomía"},
    {"id": 41, "destination": "Salento",               "category": "naturaleza","country": "Colombia",   "cost_usd": 380,  "description": "Corazón del eje cafetero con palmas de cera y cultura cafetera"},
    {"id": 42, "destination": "Barranquilla",          "category": "ciudad",    "country": "Colombia",   "cost_usd": 500,  "description": "La Arenosa con el segundo carnaval más importante del mundo"},
    {"id": 43, "destination": "Bucaramanga",           "category": "ciudad",    "country": "Colombia",   "cost_usd": 450,  "description": "Ciudad bonita de Colombia con el mejor clima y zona de aventura"},
    {"id": 44, "destination": "Mompox",                "category": "historia",  "country": "Colombia",   "cost_usd": 420,  "description": "Joya colonial a orillas del río Magdalena, patrimonio de la humanidad"},
    {"id": 45, "destination": "San Gil",               "category": "naturaleza","country": "Colombia",   "cost_usd": 400,  "description": "Capital colombiana del deporte extremo con rafting y parapente"},
    {"id": 46, "destination": "Guatapé",               "category": "naturaleza","country": "Colombia",   "cost_usd": 300,  "description": "El Peñol y pueblo zocalero con colores únicos en Antioquia"},
    {"id": 47, "destination": "Jardín",                "category": "naturaleza","country": "Colombia",   "cost_usd": 350,  "description": "Pueblo antioqueño de ensueño rodeado de montañas y cafetales"},
    {"id": 48, "destination": "Manizales",             "category": "naturaleza","country": "Colombia",   "cost_usd": 420,  "description": "Ciudad andina con Nevado del Ruiz y el mejor festival de teatro"},
    {"id": 49, "destination": "Pereira",               "category": "ciudad",    "country": "Colombia",   "cost_usd": 420,  "description": "Ciudad sin puertas del eje cafetero con termales y avistamiento de aves"},
]

DISTRACTED_DRIVING_CLASSES = [
    {"id": 0,  "name": "Conducción segura",           "name_en": "Safe driving",                     "emoji": "✅"},
    {"id": 1,  "name": "Girando / Mirando espejos",   "name_en": "Turning / Mirror checking",         "emoji": "🔄"},
    {"id": 2,  "name": "Texteando al conducir",       "name_en": "Texting on phone",                 "emoji": "📱"},
    {"id": 3,  "name": "Hablando por teléfono",       "name_en": "Talking on phone",                 "emoji": "📞"},
    {"id": 4,  "name": "Otras actividades",           "name_en": "Other distracting activities",     "emoji": "🥤"},
]

DEMAND_DESTINATIONS = ["Cartagena", "Bogotá", "Medellín", "Santa Marta", "San Andrés"]

# ─────────────────────────────────────────────
#  ROUTES - PAGES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/modulo1")
def modulo1():
    return render_template("modulo1.html", destinations=DEMAND_DESTINATIONS)


@app.route("/modulo2")
def modulo2():
    return render_template("modulo2.html", classes=DISTRACTED_DRIVING_CLASSES)


@app.route("/modulo3")
def modulo3():
    return render_template("modulo3.html")


# ─────────────────────────────────────────────
#  API ENDPOINTS
# ─────────────────────────────────────────────

@app.route("/api/predict_demand", methods=["POST"])
def predict_demand():
    """
    Simulated LSTM demand prediction.
    Returns 60-day historical + 30-day forecast time series.
    """
    data = request.get_json(silent=True) or {}
    destination = data.get("destination", "Cartagena")

    # Reproducible seed per destination
    seed_map = {d: i * 42 for i, d in enumerate(DEMAND_DESTINATIONS)}
    seed = seed_map.get(destination, 0)
    rng = np.random.default_rng(seed)

    # Base demand parameters per destination
    base_params = {
        "Cartagena":   {"base": 12000, "trend": 15,  "amplitude": 2500},
        "Bogotá":      {"base": 18000, "trend": 20,  "amplitude": 1800},
        "Medellín":    {"base": 14000, "trend": 25,  "amplitude": 2000},
        "Santa Marta": {"base": 9500,  "trend": 12,  "amplitude": 2200},
        "San Andrés":  {"base": 7500,  "trend": 8,   "amplitude": 1600},
    }
    params = base_params.get(destination, base_params["Cartagena"])

    total_days = 90  # 60 hist + 30 forecast
    t = np.arange(total_days)
    base = params["base"]
    trend = params["trend"]
    amp = params["amplitude"]

    # Sinusoidal seasonal pattern + linear trend + noise
    signal = (
        base
        + trend * t
        + amp * np.sin(2 * np.pi * t / 30)           # monthly seasonality
        + (amp * 0.4) * np.sin(2 * np.pi * t / 7)   # weekly seasonality
        + rng.normal(0, amp * 0.15, total_days)       # noise
    )
    signal = np.clip(signal, 0, None).astype(float)

    # Split into historical and forecast
    hist = signal[:60].tolist()
    fore = signal[60:].tolist()

    # Generate dates
    today = datetime.now().date()
    start_hist = today - timedelta(days=89)
    dates_hist = [(start_hist + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(60)]
    dates_fore = [(today + timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(30)]

    # Simulated error metrics
    rmse = float(rng.uniform(180, 420))
    mae  = float(rng.uniform(130, 310))
    mape = float(rng.uniform(3.5, 8.9))

    return jsonify({
        "destination":  destination,
        "historical":   [round(v, 1) for v in hist],
        "forecast":     [round(v, 1) for v in fore],
        "dates_hist":   dates_hist,
        "dates_fore":   dates_fore,
        "rmse":         round(rmse, 2),
        "mae":          round(mae, 2),
        "mape":         round(mape, 2),
    })


@app.route("/api/classify_image", methods=["POST"])
def classify_image():
    """
    Real ResNet18 image classification.
    Accepts an image upload; returns class probabilities.
    """
    if "image" not in request.files:
        return jsonify({"error": "No se encontró imagen en la solicitud."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Nombre de archivo vacío."}), 400

    try:
        # Load image via PIL
        img = Image.open(file.stream).convert('RGB')
        
        # Apply standard ResNet18 transformations
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD  = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        tensor = transform(img).unsqueeze(0) # [1, 3, 224, 224]
        
        # Inference using the real model
        real_model.eval()
        with torch.no_grad():
            outputs = real_model(tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0).numpy().tolist()
            
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        
        return jsonify({
            "predicted_class":   predicted_class,
            "class_name":        DISTRACTED_DRIVING_CLASSES[predicted_class]["name"],
            "class_name_en":     DISTRACTED_DRIVING_CLASSES[predicted_class]["name_en"],
            "emoji":             DISTRACTED_DRIVING_CLASSES[predicted_class]["emoji"],
            "confidence":        round(confidence, 4),
            "all_probabilities": [round(p, 4) for p in probs],
            "classes":           DISTRACTED_DRIVING_CLASSES,
        })
    except Exception as e:
        return jsonify({"error": f"Error al procesar la imagen: {str(e)}"}), 500


@app.route("/api/get_recommendations", methods=["POST"])
def get_recommendations():
    """
    Simulated NCF recommendation system.
    Accepts user_id (0–499); returns top-5 personalized destinations.
    """
    data = request.get_json(silent=True) or {}
    user_id = int(data.get("user_id", 0))

    if not (0 <= user_id <= 499):
        return jsonify({"error": "El user_id debe estar entre 0 y 499."}), 400

    # Reproducible per user
    rng = np.random.default_rng(seed=user_id * 37 + 13)

    # Weighted sampling — slightly favor Colombian destinations for lower user IDs
    n = len(DESTINATIONS_CATALOG)
    weights = rng.dirichlet(np.ones(n) * 0.5)
    # Bias toward first 10 (Colombian) for user_id < 250
    if user_id < 250:
        weights[:10] *= 1.8
        weights /= weights.sum()

    indices = rng.choice(n, size=5, replace=False, p=weights)
    scores_raw = rng.uniform(0.72, 0.99, size=5)
    scores_raw = np.sort(scores_raw)[::-1]  # descending

    recommendations = []
    for rank, (idx, score) in enumerate(zip(indices, scores_raw), start=1):
        dest = DESTINATIONS_CATALOG[int(idx)]
        recommendations.append({
            "rank":        rank,
            "destination": dest["destination"],
            "category":    dest["category"],
            "country":     dest["country"],
            "cost_usd":    dest["cost_usd"],
            "score":       round(float(score), 4),
            "description": dest["description"],
        })

    return jsonify({
        "user_id":        user_id,
        "recommendations": recommendations,
    })


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Dynamic port binding for production environments (e.g., Render/Heroku)
    port = int(os.environ.get("PORT", 5000))
    # Bind to 0.0.0.0 to allow external access, debug mode enabled only in development
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
