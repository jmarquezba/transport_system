"""
Reentrenamiento rápido y óptimo de los 3 modelos.
Corre: python retrain_fast.py
Tiempo estimado: ~15-20 min total
  - CNN  : ~10 min (layer4+fc, 224px, 15 épocas)
  - LSTM : ~3  min (datos de Kaggle, 150 épocas con early stop)
  - NCF  : ~2  min (60 épocas, LOO evaluation)
"""
import subprocess, sys, time
from pathlib import Path

BASE = Path(__file__).parent

scripts = [
    ("LSTM — Predicción de Demanda",    BASE/"modulo1_demanda"/"train_lstm.py"),
    ("CNN  — Conducción Distractiva",   BASE/"modulo2_clasificacion"/"train_cnn.py"),
    ("NCF  — Recomendaciones",          BASE/"modulo3_recomendacion"/"train_ncf.py"),
]

failed = []
for name, script in scripts:
    print(f"\n{'='*60}")
    print(f"  Entrenando: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    rc = subprocess.run([sys.executable, str(script)], check=False).returncode
    elapsed = time.time() - t0
    if rc == 0:
        print(f"  [OK] {name} — {elapsed/60:.1f} min")
    else:
        print(f"  [FAIL] {name} — código {rc}")
        failed.append(name)

print(f"\n{'='*60}")
if failed:
    print(f"Fallos: {failed}")
    sys.exit(1)
else:
    print("Todos los modelos entrenados. Reinicia Flask:")
    print("  cd webapp && python app.py")
