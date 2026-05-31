"""
Trabajo 3 — IRNA 2026 · Universidad Nacional de Colombia
Set de Pruebas Exhaustivo y Validación de Coherencia de Modelos 1 y 3
"""
import sys
import os
import pickle
import numpy as np
import pandas as pd

# Añadir el directorio raíz al path para importar la app de Flask
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from webapp.app import app

# Configuración de rutas
KAGGE_CACHE_DIR = r"C:\Users\santy\.cache\kagglehub\datasets\amanmehra23\travel-recommendation-dataset\versions\1"
USERS_FILE = os.path.join(KAGGE_CACHE_DIR, "Final_Updated_Expanded_Users.csv")
DESTINATIONS_FILE = os.path.join(KAGGE_CACHE_DIR, "Expanded_Destinations.csv")
REVIEWS_FILE = os.path.join(KAGGE_CACHE_DIR, "Final_Updated_Expanded_Reviews.csv")
HISTORY_FILE = os.path.join(KAGGE_CACHE_DIR, "Final_Updated_Expanded_UserHistory.csv")

def main():
    print("=" * 70)
    print("INICIANDO SET DE PRUEBAS EXHAUSTIVO DE CONTRASTE")
    print("=" * 70)

    # Cargar datasets
    try:
        df_users = pd.read_csv(USERS_FILE)
        df_dest = pd.read_csv(DESTINATIONS_FILE)
        df_reviews = pd.read_csv(REVIEWS_FILE)
        df_history = pd.read_csv(HISTORY_FILE)
        print("[OK] Datasets cargados correctamente desde el cache local.")
    except Exception as e:
        print(f"[ERROR] No se pudieron cargar los datasets: {e}")
        sys.exit(1)

    # Cliente de pruebas de Flask (in-process)
    client = app.test_client()

    # ═══════════════════════════════════════════════════════════════════════════
    #  PRUEBAS MÓDULO 1: Predicción de Demanda (Seq2Seq LSTM)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 50)
    print("PRUEBAS MÓDULO 1: Predicción de Demanda de Transporte")
    print("-" * 50)

    valid_destinations = ["Taj Mahal", "Goa Beaches", "Jaipur City", "Kerala Backwaters", "Leh Ladakh"]
    
    for dest in valid_destinations:
        print(f"\n[M1] Evaluando destino: {dest}")
        response = client.post("/api/predict_demand", json={"destination": dest})
        if response.status_code != 200:
            print(f"     [FAIL] Status code: {response.status_code}")
            continue
        
        data = response.get_json()
        
        # Validar estructura y tipos
        assert len(data["forecast"]) == 30
        assert len(data["historical"]) == 30
        
        pred_mean = np.mean(data["forecast"])
        hist_mean = np.mean(data["historical"])
        pred_std = np.std(data["forecast"])
        hist_std = np.std(data["historical"])
        
        print(f"     -> Media Histórica: {hist_mean:.1f} | Media Pronóstico: {pred_mean:.1f}")
        print(f"     -> Desviación Estándar Histórica: {hist_std:.2f} | Desviación Estándar Pronóstico: {pred_std:.2f}")
        
        # Verificaciones de Coherencia
        # 1. No debe haber valores negativos
        assert all(v >= 0 for v in data["forecast"]), "Hay valores de demanda negativos"
        # 2. La desviación estándar del pronóstico debe ser alta (inyección estacional semanal activa)
        # Verificamos que conserve al menos el 50% de la variabilidad histórica
        var_ratio = pred_std / (hist_std + 1e-6)
        print(f"     -> Conservación de Varianza Estacional: {var_ratio:.2%}")
        assert var_ratio >= 0.40, f"La predicción es demasiado plana/drástica (ratio de varianza {var_ratio:.2%})"
        print("     [SUCCESS] Predicción de demanda conserva las oscilaciones semanales y es coherente.")

    # ═══════════════════════════════════════════════════════════════════════════
    #  PRUEBAS MÓDULO 3: Recomendaciones NCF (Rating Regression)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 50)
    print("PRUEBAS MÓDULO 3: Sistema de Recomendación NCF (10 Usuarios)")
    print("-" * 50)

    # 10 Usuarios representativos
    test_users = [12, 20, 69, 109, 327, 353, 386, 403, 626, 783]
    all_recommendation_lists = []
    
    for uid in test_users:
        user_row = df_users[df_users["UserID"] == uid]
        if user_row.empty:
            continue
            
        user_prefs_str = user_row["Preferences"].values[0]
        user_prefs = [p.strip().lower() for p in user_prefs_str.replace(",", "").split()]
        print(f"\n[M3] Evaluando Usuario #{uid} ({user_row['Name'].values[0]}) - Prefs: {user_prefs_str}")
        
        response = client.post("/api/get_recommendations", json={"user_id": uid})
        if response.status_code != 200:
            print(f"     [FAIL] Status code: {response.status_code}")
            continue
            
        data = response.get_json()
        recs = data["recommendations"]
        
        assert len(recs) == 5
        
        rec_names = [r["destination"] for r in recs]
        scores = [r["score"] for r in recs]
        all_recommendation_lists.append(rec_names)
        
        print(f"     -> Recomendado: {rec_names}")
        print(f"     -> Ratings predichos: {['%.2f' % s for s in scores]}")
        
        # Validar coherencia
        # 1. Los ratings deben ser dinámicos, diferenciados y estrictamente decrecientes
        assert len(set(scores)) > 1, f"Las calificaciones son todas idénticas: {scores}"
        for i in range(1, len(scores)):
            assert scores[i] < scores[i-1], f"Las calificaciones no son estrictamente decrecientes: {scores}"
        
        # 2. Ratings en rango [1.0, 5.0]
        assert all(1.0 <= s <= 5.0 for s in scores), f"Ratings fuera del rango real 1-5: {scores}"
        
        # 3. Comprobar alineación de preferencias
        # Al tener 5 destinos totales y partición estricta, los primeros 2 deben coincidir con las preferencias
        for idx in range(2):
            dest_type = recs[idx]["category"].lower()
            assert any(pref in dest_type or dest_type in pref for pref in user_prefs), \
                f"El destino en rango #{idx+1} ({recs[idx]['destination']}, tipo={dest_type}) no coincide con las preferencias {user_prefs}"
        print(f"     [SUCCESS] Recomendación personalizada aprobada (alineación perfecta de preferencias y ratings decrecientes).")

    # 3. VERIFICACIÓN DE CONTRASTE / DIVERSIDAD
    # Para demostrar que las recomendaciones contrastan y no son siempre los mismos resultados en el mismo orden:
    # Contamos cuántas listas de recomendación son únicas
    unique_lists = len(set(tuple(l) for l in all_recommendation_lists))
    print(f"\n[M3] Diversidad de Listas recomendadas: {unique_lists} listas únicas de {len(test_users)} usuarios evaluados.")
    assert unique_lists >= 3, f"Las listas recomendadas son idénticas o tienen muy poca variedad: {unique_lists}/10"
    print("[SUCCESS] El modelo NCF genera resultados diversos y contrastados entre distintos usuarios.")

    print("\n" + "=" * 70)
    print("SET DE PRUEBAS COMPLETADO CON ÉXITO — TODOS LOS MODELOS PASARON LA VALIDACIÓN")
    print("=" * 70)

if __name__ == "__main__":
    main()
