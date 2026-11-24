from pathlib import Path
import numpy as np
import pandas as pd

# 1. Cargar el archivo comprimido generado
ruta_npz = (
    Path(__file__).resolve().parent.parent / "dataset" / "datos_procesados.npz"
)
datos = np.load(ruta_npz, allow_pickle=True)

X_train, y_train = datos["X_train"], datos["y_train"]
X_test, y_test = datos["X_test"], datos["y_test"]
clases = datos["clases"]

print("=== AUDITORÍA DEL DATASET PROCESADO ===")
print(f"Total de instancias listas: {len(y_train) + len(y_test):,}")
print(
    f"Dimensiones TRAIN: {X_train.shape} | Dimensiones TEST: {X_test.shape}"
)
print(
    f"Descriptores numéricos continuos: {X_train.shape[1]} (Tipo: {X_train.dtype})"
)
print(
    f"Valores NaN o Infinitos: {np.isnan(X_train).sum() + np.isinf(X_train).sum()}"
)
print(
    f"Verificación StandardScaler (Train) -> Media: {X_train.mean():.4f} | Desv. Estándar: {X_train.std():.4f}"
)


# 2. Tabla de balanceo en Train y Test
def resumen_clases(y_array, nombres_clases):
    conteos = pd.Series(y_array).value_counts().sort_index()
    porcentajes = (conteos / len(y_array) * 100).round(2)
    return pd.DataFrame({
        "Clase": [nombres_clases[i] for i in conteos.index],
        "Registros": conteos.values,
        "Porcentaje (%)": porcentajes.values,
    })


print("\n--- DISTRIBUCIÓN EN TRAIN ---")
print(resumen_clases(y_train, clases).to_string(index=False))

print("\n--- DISTRIBUCIÓN EN TEST ---")
print(resumen_clases(y_test, clases).to_string(index=False))