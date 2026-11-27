import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Permite importar optimization/ y evaluation/ sin importar desde dónde se ejecute
RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from evaluation.graficas import generar_todas_las_graficas
from optimization.ga_gwo import HibridoGAGWO
from optimization.mlp_evaluator import MLPEvaluator, cargar_datos


def ejecutar_experimento_oficial(
    poblaciones=(6, 12, 20),
    max_iter=10,
    eval_sample_size=30000,  # Cambia a None si quieres buscar sobre los 100,000 completos
    epocas_finales=35,
):
    carpeta_res = RAIZ / "results"
    carpeta_res.mkdir(exist_ok=True)

    X_train, y_train, X_test, y_test, clases = cargar_datos()
    evaluador = MLPEvaluator(
        X_train,
        y_train,
        X_test,
        y_test,
        clases,
        eval_sample_size=eval_sample_size,
    )

    registros_comparativa = []
    tabla_resumen = []

    mejor_corrida = None
    mejor_fit_global = -np.inf

    for n in poblaciones:
        print(f"\n==========================================")
        print(f" CORRIENDO GA-GWO | Población N = {n} | Iteraciones = {max_iter}")
        print(f"==========================================")

        opt = HibridoGAGWO(
            evaluador=evaluador,
            n_poblacion=n,
            max_iter=max_iter,
            seed=42,
        )
        res = opt.ejecutar(
            callback=lambda s: print(
                f"[N={n:02d} | Iter {s['iteracion']:02d}] "
                f"Alfa: {s['fit_alfa']:.4f} | "
                f"Media: {s['media']:.4f} | "
                f"Mediana: {s['mediana']:.4f} | "
                f"Std: {s['desv_std']:.4f}"
            )
        )

        # Guardar historial de esta población para la gráfica comparativa
        for fila in res["historial"]:
            fila_comp = fila.copy()
            fila_comp["tiempo_total_seg"] = res["tiempo_segundos"]
            registros_comparativa.append(fila_comp)

        # Estadísticas de la última iteración para la tabla del PDF
        ultima = res["historial"][-1]
        tabla_resumen.append({
            "Poblacion_N": n,
            "Mejor_F1_Alfa": round(res["mejor_fitness_val"], 4),
            "Media_Final": round(ultima["media"], 4),
            "Mediana_Final": round(ultima["mediana"], 4),
            "Desv_Std_Final": round(ultima["desv_std"], 4),
            "Tiempo_Seg": res["tiempo_segundos"],
            "Mejor_Config": str(res["mejores_hiperparametros"]),
        })

        # Conservar la mejor corrida global para entrenar y auditar el modelo definitivo
        if res["mejor_fitness_val"] > mejor_fit_global:
            mejor_fit_global = res["mejor_fitness_val"]
            mejor_corrida = (n, res)

    # 1. Guardar comparativa y tabla estadística del Punto 4
    df_comp = pd.DataFrame(registros_comparativa)
    df_comp.to_csv(carpeta_res / "comparativa_poblaciones.csv", index=False)

    df_tabla = pd.DataFrame(tabla_resumen)
    df_tabla.to_csv(carpeta_res / "resumen_estadistico_punto4.csv", index=False)

    # 2. Guardar historial de la población ganadora para las gráficas de evolución y boxplot
    n_ganador, res_ganador = mejor_corrida
    pd.DataFrame(res_ganador["historial"]).to_csv(
        carpeta_res / "historial_experimentos.csv", index=False
    )

    # 3. Entrenar modelo definitivo con los 100,000 datos y auditar sobreajuste + Test
    print(f"\n>>> Mejor población encontrada: N={n_ganador} (F1 Val={mejor_fit_global:.4f})")
    print(">>> Entrenando modelo ganador con los 100,000 datos y auditando en TEST...")
    auditoria = evaluador.entrenar_y_auditar_final(
        res_ganador["mejor_vector"], epocas=epocas_finales
    )

    resumen_json = {
        "optimizacion": {
            "n_poblacion_ganadora": n_ganador,
            "max_iter": max_iter,
            "tiempo_segundos": res_ganador["tiempo_segundos"],
            "mejor_fitness_val": res_ganador["mejor_fitness_val"],
            "mejores_hiperparametros": res_ganador["mejores_hiperparametros"],
        },
        "curvas_entrenamiento": auditoria["curvas_entrenamiento"],
        "metricas_test": auditoria["metricas_test"],
    }
    with open(carpeta_res / "mejor_modelo.json", "w", encoding="utf-8") as f:
        json.dump(resumen_json, f, indent=4, ensure_ascii=False)

    # 4. Generar automáticamente las 5 gráficas oficiales
    generar_todas_las_graficas()

    print("\n=== TABLA ESTADÍSTICA PUNTO 4 ===")
    print(df_tabla.drop(columns=["Mejor_Config"]).to_string(index=False))
    print("\n=== EVALUACIÓN FINAL EN TEST ===")
    print("Hiperparámetros óptimos:", res_ganador["mejores_hiperparametros"])
    print(f"F1-Score Macro (Test):  {auditoria['metricas_test']['f1_macro']:.4f}")
    print(f"Accuracy (Test):        {auditoria['metricas_test']['accuracy']:.4f}")


if __name__ == "__main__":
    ejecutar_experimento_oficial(
        poblaciones=(6, 12, 20),
        max_iter=10,
        eval_sample_size=30000,
        epocas_finales=35,
    )