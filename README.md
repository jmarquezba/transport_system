# 🧠 Trabajo 3 — IRNA · Universidad Nacional de Colombia

> **Inteligencia artificial para la Red Neuronal Aplicada** · Semestre 9 · 2026

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-black?logo=flask&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completo-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)
![UNAL](https://img.shields.io/badge/Universidad-Nacional%20de%20Colombia-8B0000)

---

## 📋 Descripción General

Este repositorio contiene el **Trabajo 3** de la asignatura *Redes Neuronales Aplicadas (IRNA)* de la **Universidad Nacional de Colombia**. El proyecto integra tres módulos independientes de Machine Learning orientados a problemáticas reales del sector turístico y de movilidad en Colombia, junto con una herramienta web interactiva que unifica los modelos entrenados.

| Módulo | Tarea | Modelo | Dominio |
|--------|-------|--------|---------|
| 1 | Predicción de demanda de transporte | LSTM | Turismo / Series de tiempo |
| 2 | Clasificación de comportamiento del conductor | ResNet18 (Transfer Learning) | Seguridad vial / Visión por computadora |
| 3 | Sistema de recomendación de viajes | Neural Collaborative Filtering (NCF) | Turismo / Sistemas de recomendación |

---

## 📁 Estructura del Proyecto

```
Trabajo3_IRNA/
│
├── 📓 Módulo 1 — LSTM Predicción de Demanda
│   └── Modulo1_LSTM_Demanda.ipynb
│
├── 📓 Módulo 2 — ResNet18 Clasificación Conductores
│   └── Modulo2_ResNet18_Conductores.ipynb
│
├── 📓 Módulo 3 — NCF Sistema de Recomendación
│   └── Modulo3_NCF_Recomendacion.ipynb
│
├── 🌐 Herramienta Web (Flask)
│   ├── app.py                         # Servidor Flask principal
│   ├── templates/
│   │   ├── index.html                 # Página de inicio
│   │   ├── modulo1.html               # UI Predicción de demanda
│   │   ├── modulo2.html               # UI Clasificación conductor
│   │   └── modulo3.html               # UI Recomendación de viajes
│   └── static/
│       ├── css/
│       ├── js/
│       └── img/
│
├── 📦 Modelos entrenados
│   ├── modelos/
│   │   ├── lstm_demanda_{destino}.pt
│   │   ├── resnet18_conductores.pt
│   │   └── ncf_recomendacion.pt
│   └── scalers/
│       └── scaler_{destino}.pkl
│
├── 📊 Resultados y Gráficas
│   └── resultados/
│       ├── modulo1/
│       ├── modulo2/
│       └── modulo3/
│
├── 📄 informe_tecnico.md              # Informe técnico completo
├── 📄 canvas_design_thinking.md       # Canvas de Design Thinking y BMC
├── 📄 README.md                       # Este archivo
└── 📄 requirements.txt                # Dependencias Python
```

---

## ⚙️ Instalación

### Prerrequisitos

- Python **3.10** o superior
- pip **23+**
- (Recomendado) GPU con CUDA para entrenamiento

### Pasos de instalación

```bash
# 1. Clonar o descargar el repositorio
git clone <URL_DEL_REPOSITORIO>
cd Trabajo3_IRNA

# 2. Crear entorno virtual (recomendado)
python -m venv venv

# En Windows:
venv\Scripts\activate

# En Linux/macOS:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar instalación de PyTorch
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

> **Nota:** Si tienes GPU NVIDIA, instala la versión de PyTorch con soporte CUDA desde [pytorch.org](https://pytorch.org/get-started/locally/).

---

## 🚀 Cómo Ejecutar los Notebooks

Cada módulo es independiente y puede ejecutarse por separado. Se recomienda ejecutar las celdas en orden desde la primera hasta la última.

### Módulo 1 — Predicción de Demanda LSTM

```bash
jupyter notebook Modulo1_LSTM_Demanda.ipynb
```

**Qué hace:**
1. Genera datos sintéticos de demanda turística (15,000 registros, 10 destinos colombianos)
2. Aplica preprocesamiento con MinMaxScaler
3. Construye ventanas deslizantes de 30 días
4. Entrena un LSTM de 2 capas (64 unidades ocultas) con Early Stopping
5. Evalúa con RMSE, MAE y MAPE por destino
6. Genera pronóstico autorregresivo a 30 días
7. Guarda modelos y scalers en `modelos/`

**Tiempo estimado de entrenamiento:** ~5–10 min (CPU) / ~1–2 min (GPU)

---

### Módulo 2 — Clasificación CNN ResNet18

```bash
jupyter notebook Modulo2_ResNet18_Conductores.ipynb
```

**Qué hace:**
1. Genera/descarga imágenes sintéticas de conducción distraída (2,000 imágenes, 10 clases)
2. Aplica data augmentation (rotación, flip, color jitter, normalización ImageNet)
3. Carga ResNet18 preentrenado en ImageNet
4. Congela capas `conv1` a `layer3`, fine-tune de `layer4` + cabeza FC
5. Entrena con CrossEntropyLoss y Adam
6. Evalúa: Accuracy, F1-macro, Precision, Recall, Matriz de Confusión
7. Guarda modelo en `modelos/resnet18_conductores.pt`

**Tiempo estimado de entrenamiento:** ~15–25 min (CPU) / ~3–5 min (GPU)

---

### Módulo 3 — Sistema de Recomendación NCF

```bash
jupyter notebook Modulo3_NCF_Recomendacion.ipynb
```

**Qué hace:**
1. Genera datos sintéticos: 500 usuarios, 50 destinos latinoamericanos, ~8,000 interacciones
2. Aplica negative sampling (ratio 4:1) y split 80/20
3. Construye arquitectura NCF (embedding + MLP)
4. Entrena con Binary Cross-Entropy y Adam
5. Evalúa: Precision@5, Recall@5, NDCG@5, NDCG@10
6. Visualiza embeddings de destinos con PCA 2D/3D
7. Guarda modelo en `modelos/ncf_recomendacion.pt`

**Tiempo estimado de entrenamiento:** ~3–8 min (CPU) / ~1–2 min (GPU)

---

## 🌐 Cómo Ejecutar la Herramienta Web

La aplicación web integra los tres módulos en una interfaz unificada basada en Flask.

```bash
# Asegúrate de que los modelos ya estén entrenados (ejecutar notebooks primero)

# Iniciar el servidor Flask
python app.py
```

Luego abre tu navegador en:

```
http://localhost:5000
```

### Rutas disponibles

| Ruta | Descripción |
|------|-------------|
| `/` | Página principal con descripción del proyecto |
| `/modulo1` | Interfaz de predicción de demanda turística |
| `/modulo2` | Clasificador de comportamiento del conductor |
| `/modulo3` | Sistema de recomendación de viajes |
| `/api/predict_demand` | API REST - predicción de demanda (POST) |
| `/api/classify_driver` | API REST - clasificación conductor (POST) |
| `/api/recommend` | API REST - recomendaciones de viaje (POST) |

---

## 📊 Descripción de los Módulos

### 🔵 Módulo 1: Predicción de Demanda de Transporte

**Objetivo:** Predecir la demanda turística diaria para destinos colombianos clave, ayudando a operadores de transporte a optimizar la asignación de recursos.

| Parámetro | Valor |
|-----------|-------|
| Registros | 15,000 (1,500 por destino) |
| Destinos | 10 ciudades colombianas |
| Lookback window | 30 días |
| Capas LSTM | 2 |
| Unidades ocultas | 64 |
| Épocas máx. | 100 |
| Early stopping patience | 10 |
| Preprocesamiento | MinMaxScaler |

**Destinos incluidos:** Cartagena, Bogotá, Medellín, Santa Marta, San Andrés, Cali, Villa de Leyva, Salento, Leticia, Bucaramanga

---

### 🟠 Módulo 2: Clasificación de Comportamiento del Conductor

**Objetivo:** Detectar automáticamente comportamientos de conducción distraída usando visión por computadora, para apoyar sistemas de seguridad activa en vehículos.

| Parámetro | Valor |
|-----------|-------|
| Imágenes totales | 2,000 |
| Clases | 10 |
| Imágenes por clase | 200 |
| Arquitectura base | ResNet18 (ImageNet) |
| Capas fine-tuned | layer4 + FC head |
| Capas congeladas | conv1, layer1, layer2, layer3 |
| Tamaño de imagen | 224×224 |

**Clases de distracción:**
1. Conducción segura
2. Enviando mensajes (mano derecha)
3. Hablando por teléfono (mano derecha)
4. Enviando mensajes (mano izquierda)
5. Hablando por teléfono (mano izquierda)
6. Operando radio
7. Bebiendo
8. Alcanzando hacia atrás
9. Maquillándose
10. Hablando con pasajero

---

### 🟢 Módulo 3: Sistema de Recomendación de Viajes

**Objetivo:** Recomendar destinos turísticos personalizados usando Filtrado Colaborativo Neuronal, modelando preferencias implícitas de usuarios.

| Parámetro | Valor |
|-----------|-------|
| Usuarios | 500 |
| Destinos | 50 (Latinoamérica) |
| Interacciones | ~8,000 |
| Negative sampling ratio | 4:1 |
| Train / Test split | 80% / 20% |
| Dimensión embedding | 32 |
| Capas MLP | [64, 32, 16] |

**Métricas de evaluación:** Precision@5, Recall@5, NDCG@5, NDCG@10

---

## 👥 Equipo y Curso

| Campo | Información |
|-------|-------------|
| **Asignatura** | Redes Neuronales Aplicadas (IRNA) |
| **Universidad** | Universidad Nacional de Colombia |
| **Semestre** | 9 — 2026 |
| **Trabajo** | Trabajo 3 — Módulos integrados con herramienta web |

---

## 📖 Referencias Rápidas

- [PyTorch Documentation](https://pytorch.org/docs/)
- [ResNet Paper (He et al., 2016)](https://arxiv.org/abs/1512.03385)
- [LSTM Paper (Hochreiter & Schmidhuber, 1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [NCF Paper (He et al., 2017)](https://arxiv.org/abs/1708.05031)
- [State Farm Distracted Driver Dataset](https://www.kaggle.com/c/state-farm-distracted-driver-detection)

---

## 📄 Licencia

Este proyecto fue desarrollado con fines académicos en la **Universidad Nacional de Colombia**. El código es de libre uso para propósitos educativos.

---

<p align="center">
  <strong>Universidad Nacional de Colombia · IRNA · 2026</strong><br>
  <em>Redes Neuronales Aplicadas al Turismo y la Seguridad Vial</em>
</p>
