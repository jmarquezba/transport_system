#!/usr/bin/env bash
# ============================================================
# train_and_commit.sh
# Corre este script UNA VEZ en tu máquina local.
# Entrena los 3 modelos con datos reales de Kaggle,
# luego hace git add + commit de los .pt y .pkl
# para que el deployment no necesite reentrenar.
# ============================================================
set -e

echo "=== [1/4] Instalando dependencias ==="
pip install -r requirements.txt

echo "=== [2/4] Entrenando todos los módulos ==="
python train_all.py --force

echo "=== [3/4] Verificando artefactos generados ==="
python -c "
from pathlib import Path
models = Path('webapp/models')
required = ['lstm_demanda.pt','scaler_demanda.pkl','routes_metadata.pkl',
            'resnet18_driver.pt','class_names.pkl',
            'ncf_model.pt','ncf_metadata.pkl']
missing = [f for f in required if not (models/f).exists()]
if missing:
    print('FALTAN:', missing); exit(1)
else:
    for f in required:
        size = (models/f).stat().st_size
        print(f'  OK  {f}  ({size/1024:.0f} KB)')
    print('Todos los modelos presentes.')
"

echo "=== [4/4] Commiteando modelos pre-entrenados ==="
git add webapp/models/
git commit -m "feat: add pre-trained models (LSTM, ResNet18, NCF) - real Kaggle data"

echo ""
echo "Listo. Haz git push para que el deployment use los modelos pre-entrenados."
echo "El servidor ya no necesita entrenar al arrancar."
