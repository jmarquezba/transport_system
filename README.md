# Trabajo 3 — Sistema Inteligente Integrado para Empresa de Transporte
### IRNA 2026-01 · Universidad Nacional de Colombia · Sede Medellín

---

## Descripción
Sistema de Deep Learning con tres módulos:
1. **Módulo 1** — Predicción de demanda de transporte (LSTM)
2. **Módulo 2** — Clasificación de conducción distractiva (ResNet18)
3. **Módulo 3** — Recomendación de destinos de viaje (NCF)

**Todos los modelos usan datos reales de Kaggle. Sin datos sintéticos.**

---

## Datasets de Kaggle (obligatorios)
| Módulo | Dataset |
|--------|---------|
| 1 y 3  | `amanmehra23/travel-recommendation-dataset` |
| 2      | `arafatsahinafridi/multi-class-driver-behavior-image-dataset` |

---

## Configuración de Kaggle

### 1. Crear cuenta en Kaggle y obtener API key
Ve a https://www.kaggle.com/settings → API → Create New Token → descarga `kaggle.json`

### 2. Colocar el archivo
```
# Linux/Mac
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Windows
mkdir %USERPROFILE%\.kaggle
copy kaggle.json %USERPROFILE%\.kaggle\kaggle.json
```

### 3. En deployment (Render/Railway)
Agrega variables de entorno:
```
KAGGLE_USERNAME = tu_usuario
KAGGLE_KEY      = tu_api_key
```

---

## Instalación local

```bash
# 1. Clonar repositorio
git clone <repo_url>
cd Trabajo3_IRNA

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Entrenar todos los modelos con datos reales
python train_all.py

# 5. Iniciar la aplicación web
cd webapp
python app.py
# Abre: http://localhost:5000
```

---

## Entrenamiento por módulo

```bash
# Entrenar solo LSTM (Módulo 1)
python modulo1_demanda/train_lstm.py

# Entrenar solo CNN (Módulo 2)
python modulo2_clasificacion/train_cnn.py

# Entrenar solo NCF (Módulo 3)
python modulo3_recomendacion/train_ncf.py

# Entrenar todo, forzar re-entrenamiento
python train_all.py --force

# Entrenar solo un módulo específico
python train_all.py --module 2
```

---

## Deployment en Render

1. Fork este repositorio en GitHub
2. En Render → New Web Service → conectar repo
3. Agrega variables de entorno: `KAGGLE_USERNAME` y `KAGGLE_KEY`
4. Build Command: `bash build.sh`
5. Start Command: `cd webapp && gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`

---

## Estructura del proyecto

```
Trabajo3_IRNA/
├── modulo1_demanda/
│   ├── train_lstm.py          ← Entrenamiento LSTM (datos reales)
│   ├── 01_EDA_demanda.ipynb
│   └── 02_LSTM_demanda.ipynb
├── modulo2_clasificacion/
│   ├── train_cnn.py           ← Entrenamiento ResNet18 (datos reales)
│   ├── 01_EDA_imagenes.ipynb
│   └── 02_CNN_driver.ipynb
├── modulo3_recomendacion/
│   ├── train_ncf.py           ← Entrenamiento NCF (datos reales)
│   ├── 01_EDA_recomendacion.ipynb
│   └── 02_NCF_recomendacion.ipynb
├── webapp/
│   ├── app.py                 ← Flask app (carga modelos dinámicamente)
│   ├── models/                ← Modelos entrenados (.pt + .pkl)
│   └── templates/
├── train_all.py               ← Script maestro de entrenamiento
├── build.sh                   ← Script de build (local + deployment)
├── Procfile                   ← Para Heroku/Railway
├── render.yaml                ← Para Render
└── requirements.txt
```
