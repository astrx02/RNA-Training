import ast
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_RES = RAIZ / "results"
CARPETA_PLOTS = CARPETA_RES / "plots"
CARPETA_PLOTS.mkdir(parents=True, exist_ok=True)


def graficar_anti_sobreajuste(datos_json):
    """Genera curvas de F1-Score y Log-Loss homogéneas (Train vs Validación real) por época."""
    curvas = datos_json["curvas_entrenamiento"]
    epocas = curvas["epocas"]
    mejor_epoca = curvas.get("mejor_epoca", epocas[-1])
    idx_mejor = min(max(mejor_epoca - 1, 0), len(epocas) - 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 1. F1-Score por época
    ax1.plot(epocas, curvas["train_f1"], label="Entrenamiento (Train)", linewidth=2)
    ax1.plot(epocas, curvas["val_f1"], label="Validación Real (Val)", linewidth=2, linestyle="--")
    ax1.axvline(
        mejor_epoca,
        color="green",
        linestyle=":",
        linewidth=1.8,
        label=f"Mejor Checkpoint (Época {mejor_epoca})",
    )
    ax1.set_title("Generalización: F1-Score por Época")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("F1-Score Macro")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Pérdida (Log-Loss post-época) por época
    ax2.plot(epocas, curvas["train_loss"], label="Entrenamiento (Train)", linewidth=2)
    ax2.plot(epocas, curvas["val_loss"], label="Validación Real (Val)", linewidth=2, linestyle="--")
    ax2.axvline(
        mejor_epoca,
        color="green",
        linestyle=":",
        linewidth=1.8,
        label=f"Mejor Checkpoint (Época {mejor_epoca})",
    )
    ax2.set_title("Control de Sobreajuste: Log-Loss por Época")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Log-Loss (Entropía Cruzada)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    brecha = abs(curvas["train_f1"][idx_mejor] - curvas["val_f1"][idx_mejor])
    fig.suptitle(
        f"Diagnóstico de Sobreajuste (Sin Data Leakage) | Checkpoint Época {mejor_epoca} | Brecha Train-Val: {brecha:.4f}",
        fontsize=11.5,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(CARPETA_PLOTS / "1_anti_sobreajuste_epocas.png", dpi=300)
    plt.close()


def graficar_matriz_confusion(datos_json):
    """Genera la matriz de confusión sobre el conjunto TEST intacto (100% datos reales)."""
    metricas = datos_json["metricas_test"]
    cm = np.array(metricas["matriz_confusion"])
    clases = metricas["clases"]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(clases)),
        yticks=np.arange(len(clases)),
        xticklabels=clases,
        yticklabels=clases,
        ylabel="Etiqueta Real",
        xlabel="Etiqueta Predicha",
        title=f"Matriz de Confusión en Test Real (F1 = {metricas['f1_macro']:.4f})",
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    umbral = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > umbral else "black",
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(CARPETA_PLOTS / "2_matriz_confusion_test.png", dpi=300)
    plt.close()


def graficar_evolucion_y_tendencia(df_hist):
    """Genera evolución de GA-GWO por iteración y diagrama de cajas (tendencia central)."""
    iters = df_hist["iteracion"]

    # Gráfica 3: Evolución del líder Alfa, Media, Mediana y banda de desviación estándar acotada a [0, 1]
    fig, ax = plt.subplots(figsize=(9, 5))
    banda_inf = np.clip(df_hist["media"] - df_hist["desv_std"], 0.0, 1.0)
    banda_sup = np.clip(df_hist["media"] + df_hist["desv_std"], 0.0, 1.0)

    ax.plot(iters, df_hist["fit_alfa"], marker="o", linewidth=2.2, label="Mejor Líder (Alfa)")
    ax.plot(iters, df_hist["media"], marker="s", linestyle="--", label="Media Población")
    ax.plot(iters, df_hist["mediana"], marker="^", linestyle=":", label="Mediana Población")
    ax.fill_between(
        iters,
        banda_inf,
        banda_sup,
        alpha=0.18,
        label="Media ± Desv. Estándar",
    )
    ax.set_title("Evolución del Algoritmo Híbrido GA-GWO por Iteración")
    ax.set_xlabel("Iteración (Generación)")
    ax.set_ylabel("F1-Score Macro (Validación Real)")
    ax.set_xticks(iters)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(CARPETA_PLOTS / "3_evolucion_ga_gwo.png", dpi=300)
    plt.close()

    # Gráfica 4: Boxplot por iteración (dispersión y tendencia central)
    listas_fit = [
        ast.literal_eval(val) if isinstance(val, str) else val
        for val in df_hist["fitness_poblacion"]
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(listas_fit, tick_labels=iters, patch_artist=True)
    ax.set_title("Distribución y Tendencia Central de la Población por Iteración")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("F1-Score Macro (Validación Real)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(CARPETA_PLOTS / "4_boxplot_tendencia_central.png", dpi=300)
    plt.close()


def graficar_comparativa_poblaciones():
    """Si existe comparativa_poblaciones.csv, compara la convergencia por tamaño de población."""
    ruta_comp = CARPETA_RES / "comparativa_poblaciones.csv"
    if not ruta_comp.exists():
        return

    df_comp = pd.read_csv(ruta_comp)
    fig, ax = plt.subplots(figsize=(9, 5))

    for n_pob, grupo in df_comp.groupby("n_poblacion"):
        ax.plot(
            grupo["iteracion"],
            grupo["fit_alfa"],
            marker="o",
            linewidth=2,
            label=f"Población N = {n_pob}",
        )

    ax.set_title("Impacto del Tamaño de Población (GA) en la Convergencia")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Mejor F1-Score Macro (Alfa en Validación Real)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(CARPETA_PLOTS / "5_comparativa_poblaciones_ga.png", dpi=300)
    plt.close()


def generar_todas_las_graficas():
    with open(CARPETA_RES / "mejor_modelo.json", "r", encoding="utf-8") as f:
        datos_json = json.load(f)
    df_hist = pd.read_csv(CARPETA_RES / "historial_experimentos.csv")

    graficar_anti_sobreajuste(datos_json)
    graficar_matriz_confusion(datos_json)
    graficar_evolucion_y_tendencia(df_hist)
    graficar_comparativa_poblaciones()

    print(f"Gráficas generadas correctamente en: {CARPETA_PLOTS}")


if __name__ == "__main__":
    generar_todas_las_graficas()
