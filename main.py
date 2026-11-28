import json
from pathlib import Path
import pandas as pd
from evaluation.graficas import generar_todas_las_graficas
from optimization.ga_gwo import HibridoGAGWO
from optimization.mlp_evaluator import MLPEvaluator, cargar_datos


def imprimir_progreso(stats):
    """Ejemplo de callback para consola o interfaz."""
    print(
        f"[Iter {stats['iteracion']:02d} | a={stats['parametro_a']:.2f}] "
        f"Alfa: {stats['fit_alfa']:.4f} | Beta: {stats['fit_beta']:.4f} | Delta: {stats['fit_delta']:.4f} | "
        f"Media: {stats['media']:.4f} | Mediana: {stats['mediana']:.4f} | Std: {stats['desv_std']:.4f}"
    )


def ejecutar_pipeline(
    n_poblacion=12,
    max_iter=10,
    eval_sample_size=24000,
    epocas_finales=40,
    paciencia=8,
    callback=imprimir_progreso,
    generar_graficas=True,
):
    carpeta_res = Path(__file__).resolve().parent / "results"
    carpeta_res.mkdir(exist_ok=True)

    # 1. Cargar datos procesados (Train balanceado + Val real + Test real, 0% fuga)
    X_train, y_train, X_val, y_val, X_test, y_test, clases = cargar_datos()
    evaluador = MLPEvaluator(
        X_train,
        y_train,
        X_val,
        y_val,
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

    # 3. Entrenar modelo ganador sobre todo X_train con Early Stopping y auditar en TEST real
    print(
        f"\nEntrenando modelo ganador sobre {len(X_train):,} registros de Train "
        f"y auditando en {len(X_val):,} Val / {len(X_test):,} Test reales..."
    )
    auditoria = evaluador.entrenar_y_auditar_final(
        resultado_opt["mejor_vector"],
        epocas=epocas_finales,
        paciencia=paciencia,
    )

    # 4. Guardar resultados estructurados con esquema consistente con experiment.py
    df_hist = pd.DataFrame(resultado_opt["historial"])
    df_hist.to_csv(carpeta_res / "historial_experimentos.csv", index=False)

    resumen_json = {
        "optimizacion": {
            "n_poblacion_ganadora": n_poblacion,
            "max_iter": max_iter,
            "eval_sample_size": eval_sample_size,
            "tiempo_segundos": resultado_opt["tiempo_segundos"],
            "mejor_fitness_val": resultado_opt["mejor_fitness_val"],
            "mejores_hiperparametros": resultado_opt["mejores_hiperparametros"],
        },
        "curvas_entrenamiento": auditoria["curvas_entrenamiento"],
        "metricas_test": auditoria["metricas_test"],
    }
    with open(carpeta_res / "mejor_modelo.json", "w", encoding="utf-8") as f:
        json.dump(resumen_json, f, indent=4, ensure_ascii=False)

    if generar_graficas:
        generar_todas_las_graficas()

    curvas = auditoria["curvas_entrenamiento"]
    print("\n=== RESULTADOS FINALES EN TEST (100% REAL, SIN FUGAS) ===")
    print("Hiperparámetros óptimos:", resultado_opt["mejores_hiperparametros"])
    print(
        f"Mejor Checkpoint: Época {curvas['mejor_epoca']} "
        f"(Early Stopping: {curvas['detenido_por_early_stopping']}) | "
        f"Brecha Train-Val F1: {curvas['brecha_f1_mejor_epoca']:.4f}"
    )
    print(f"F1-Score Macro (Test):  {auditoria['metricas_test']['f1_macro']:.4f}")
    print(f"Accuracy (Test):        {auditoria['metricas_test']['accuracy']:.4f}")
    print(f"Archivos guardados en:  {carpeta_res}")
    return resumen_json


if __name__ == "__main__":
    ejecutar_pipeline(
        n_poblacion=12,
        max_iter=10,
        eval_sample_size=24000,
        epocas_finales=40,
        paciencia=8,
    )
