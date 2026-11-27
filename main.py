import json
from pathlib import Path
import pandas as pd
from optimization.ga_gwo import HibridoGAGWO
from optimization.mlp_evaluator import MLPEvaluator, cargar_datos


def imprimir_progreso(stats):
    """Ejemplo de callback que tu compañero reemplazará por su barra/log de interfaz."""
    print(
        f"[Iter {stats['iteracion']:02d}] "
        f"Alfa: {stats['fit_alfa']:.4f} | "
        f"Media: {stats['media']:.4f} | "
        f"Mediana: {stats['mediana']:.4f} | "
        f"Std: {stats['desv_std']:.4f} | "
        f"Config: {stats['mejores_hiperparametros']}"
    )


def ejecutar_pipeline(
    n_poblacion=10,
    max_iter=10,
    eval_sample_size=25000,
    epocas_finales=35,
    callback=imprimir_progreso,
):
    carpeta_res = Path(__file__).resolve().parent / "results"
    carpeta_res.mkdir(exist_ok=True)

    # 1. Cargar datos procesados
    X_train, y_train, X_test, y_test, clases = cargar_datos()
    evaluador = MLPEvaluator(
        X_train,
        y_train,
        X_test,
        y_test,
        clases,
        eval_sample_size=eval_sample_size,
    )

    # 2. Correr algoritmo híbrido GA + GWO
    print(f"Iniciando GA-GWO (Población={n_poblacion}, Iteraciones={max_iter})...")
    optimizador = HibridoGAGWO(
        evaluador=evaluador,
        n_poblacion=n_poblacion,
        max_iter=max_iter,
    )
    resultado_opt = optimizador.ejecutar(callback=callback)

    # 3. Entrenar modelo ganador sobre los 100,000 datos y auditar sobreajuste
    print("\nEntrenando modelo ganador época por época y evaluando en TEST...")
    auditoria = evaluador.entrenar_y_auditar_final(
        resultado_opt["mejor_vector"], epocas=epocas_finales
    )

    # 4. Guardar resultados estructurados para las gráficas y la interfaz
    df_hist = pd.DataFrame(resultado_opt["historial"])
    df_hist.to_csv(carpeta_res / "historial_experimentos.csv", index=False)

    resumen_json = {
        "optimizacion": {
            "n_poblacion": n_poblacion,
            "max_iter": max_iter,
            "tiempo_segundos": resultado_opt["tiempo_segundos"],
            "mejor_fitness_val": resultado_opt["mejor_fitness_val"],
            "mejores_hiperparametros": resultado_opt["mejores_hiperparametros"],
        },
        "curvas_entrenamiento": auditoria["curvas_entrenamiento"],
        "metricas_test": auditoria["metricas_test"],
    }
    with open(carpeta_res / "mejor_modelo.json", "w", encoding="utf-8") as f:
        json.dump(resumen_json, f, indent=4, ensure_ascii=False)

    print("\n=== RESULTADOS FINALES EN TEST ===")
    print("Hiperparámetros óptimos:", resultado_opt["mejores_hiperparametros"])
    print(f"F1-Score Macro (Test):  {auditoria['metricas_test']['f1_macro']:.4f}")
    print(f"Accuracy (Test):        {auditoria['metricas_test']['accuracy']:.4f}")
    print(f"Archivos guardados en:  {carpeta_res}")
    return resumen_json


if __name__ == "__main__":
    ejecutar_pipeline(
        n_poblacion=10,
        max_iter=10,
        eval_sample_size=25000,  # Cambia a None para la corrida final con los 100,000
        epocas_finales=35,
    )