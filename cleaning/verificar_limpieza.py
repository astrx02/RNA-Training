from pathlib import Path
import numpy as np
import pandas as pd

# 1. Cargar el archivo comprimido generado
ruta_npz = (
    Path(__file__).resolve().parent.parent / "dataset" / "datos_procesados.npz"
)
datos = np.load(ruta_npz, allow_pickle=True)

X_train, y_train = datos["X_train"], datos["y_train"]
X_val, y_val = datos["X_val"], datos["y_val"]
X_test, y_test = datos["X_test"], datos["y_test"]
clases = datos["clases"]


def _hashes_filas(arr):
    arr_c = np.ascontiguousarray(np.round(arr, 8))
    vista = arr_c.view(np.dtype((np.void, arr_c.dtype.itemsize * arr_c.shape[1]))).ravel()
    return set(pd.util.hash_array(vista))


def _contar_nan_inf(arr):
    return int(np.isnan(arr).sum() + np.isinf(arr).sum())


print("=== AUDITORÍA DEL DATASET PROCESADO ===")
print(f"Total de instancias procesadas: {len(y_train) + len(y_val) + len(y_test):,}")
print(
    f"Dimensiones -> TRAIN: {X_train.shape} | VAL: {X_val.shape} | TEST: {X_test.shape}"
)
print(
    f"Descriptores numéricos continuos: {X_train.shape[1]} (Tipo: {X_train.dtype})"
)
print(
    "Valores NaN o Infinitos -> "
    f"Train: {_contar_nan_inf(X_train)} | "
    f"Val: {_contar_nan_inf(X_val)} | "
    f"Test: {_contar_nan_inf(X_test)}"
)
print(
    f"Columnas constantes (std=0) en Train: {int(np.sum(X_train.std(axis=0) == 0))}"
)
print(
    f"Verificación StandardScaler (Train) -> Media: {X_train.mean():.4f} | Desv. Estándar: {X_train.std():.4f}"
)

# 2. Auditoría de Fuga de Datos (intersección exacta de filas entre particiones)
h_train = _hashes_filas(X_train)
h_val = _hashes_filas(X_val)
h_test = _hashes_filas(X_test)

ov_tr_val = len(h_train & h_val)
ov_tr_test = len(h_train & h_test)
ov_val_test = len(h_val & h_test)

print("\n--- AUDITORÍA DE FUGA DE DATOS (DATA LEAKAGE) ---")
print(f"Filas compartidas Train ∩ Val:  {ov_tr_val} ({ov_tr_val / len(X_val) * 100:.2f}%)")
print(f"Filas compartidas Train ∩ Test: {ov_tr_test} ({ov_tr_test / len(X_test) * 100:.2f}%)")
print(f"Filas compartidas Val ∩ Test:   {ov_val_test} ({ov_val_test / len(X_test) * 100:.2f}%)")


# 3. Tablas de balanceo en Train, Val y Test
def resumen_clases(y_array, nombres_clases):
    conteos = pd.Series(y_array).value_counts().sort_index()
    porcentajes = (conteos / len(y_array) * 100).round(2)
    return pd.DataFrame({
        "Clase": [nombres_clases[i] for i in conteos.index],
        "Registros": conteos.values,
        "Porcentaje (%)": porcentajes.values,
    })


print("\n--- DISTRIBUCIÓN EN TRAIN (Balanceado con RUS + SMOTE) ---")
print(resumen_clases(y_train, clases).to_string(index=False))

print("\n--- DISTRIBUCIÓN EN VAL (100% Datos Reales) ---")
print(resumen_clases(y_val, clases).to_string(index=False))

print("\n--- DISTRIBUCIÓN EN TEST (100% Datos Reales) ---")
print(resumen_clases(y_test, clases).to_string(index=False))
