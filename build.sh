#!/usr/bin/env bash
# build.sh — para Render / Railway / Heroku
# Si los modelos ya existen en webapp/models/ (commiteados al repo),
# solo instala dependencias y arranca. Sin re-entrenar.
set -e

echo "=== Instalando dependencias ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Verificando modelos pre-entrenados ==="
python train_all.py   # skip automático si ya existen los .pt

echo "=== Build completado ==="
