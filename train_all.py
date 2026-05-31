"""
Script maestro de entrenamiento.
Corre los tres módulos con datos REALES de Kaggle.
Los modelos quedan en webapp/models/ — commitéalos para que el deploy no reentrene.

Uso:
    python train_all.py              # solo lo que falta
    python train_all.py --force      # re-entrena todo
    python train_all.py --module 1   # solo LSTM
    python train_all.py --module 2   # solo CNN
    python train_all.py --module 3   # solo NCF
"""

import sys, subprocess, argparse
from pathlib import Path

BASE   = Path(__file__).resolve().parent
WEBAPP = BASE / "webapp" / "models"

SCRIPTS = {
    1: BASE / "modulo1_demanda"       / "train_lstm.py",
    2: BASE / "modulo2_clasificacion" / "train_cnn.py",
    3: BASE / "modulo3_recomendacion" / "train_ncf.py",
}

# Archivos que demuestran que el módulo ya fue entrenado
ARTIFACTS = {
    1: [WEBAPP/"lstm_demanda.pt", WEBAPP/"scaler_demanda.pkl", WEBAPP/"routes_metadata.pkl"],
    2: [WEBAPP/"resnet18_driver.pt", WEBAPP/"class_names.pkl"],
    3: [WEBAPP/"ncf_model.pt", WEBAPP/"ncf_metadata.pkl"],
}

MIN_SIZES = {1: 50_000, 2: 1_000_000, 3: 10_000}   # bytes mínimos para considerar válido


def is_trained(m: int) -> bool:
    return all(p.exists() and p.stat().st_size >= MIN_SIZES.get(m, 100) for p in ARTIFACTS[m])


def run(m: int) -> int:
    print(f"\n{'='*60}\nMÓDULO {m} — {SCRIPTS[m].name}\n{'='*60}")
    return subprocess.run([sys.executable, str(SCRIPTS[m])], check=False).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force",  action="store_true")
    ap.add_argument("--module", type=int, choices=[1,2,3])
    args = ap.parse_args()

    WEBAPP.mkdir(parents=True, exist_ok=True)
    modules = [args.module] if args.module else [1, 2, 3]
    failed  = []

    for m in modules:
        if not args.force and is_trained(m):
            print(f"[SKIP] Módulo {m}: modelos ya presentes en webapp/models/")
            continue
        if run(m) != 0:
            failed.append(m)

    if failed:
        print(f"\n[FALLO] Módulos fallidos: {failed}"); sys.exit(1)
    print(f"\n[OK] Entrenamiento completado: {modules}")


if __name__ == "__main__":
    main()
