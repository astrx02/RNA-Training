from pathlib import Path
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Cargar dataset
carpeta_dataset = Path(__file__).resolve().parent.parent / "dataset"
archivo_csv = carpeta_dataset / "RT_IOT2022"
if not archivo_csv.exists():
    archivo_csv = [f for f in carpeta_dataset.rglob("*RT*") if f.is_file()][0]
df = pd.read_csv(archivo_csv)
n_original = len(df)

# 2. Eliminar columnas no numéricas, índices residuales y columnas constantes (varianza 0)
cols_a_eliminar = [
    c for c in ["Unnamed: 0", "no", "proto", "service"] if c in df.columns
]
df = df.drop(columns=cols_a_eliminar)

# 3. Agrupar en 4 macro-clases equilibrables
mapeo_clases = {
    "DOS_SYN_Hping": "DoS_DDoS",
    "DDOS_Slowloris": "DoS_DDoS",
    "Thing_Speak": "Normal_IoT",
    "MQTT_Publish": "Normal_IoT",
    "Wipro_bulb": "Normal_IoT",
    "ARP_poisioning": "ARP_Spoofing",
    "NMAP_UDP_SCAN": "Recon_BruteForce",
    "NMAP_XMAS_TREE_SCAN": "Recon_BruteForce",
    "NMAP_OS_DETECTION": "Recon_BruteForce",
    "NMAP_TCP_scan": "Recon_BruteForce",
    "NMAP_FIN_SCAN": "Recon_BruteForce",
    "Metasploit_Brute_Force_SSH": "Recon_BruteForce",
}
df["Clase"] = df["Attack_type"].map(mapeo_clases)
df = df.dropna(subset=["Clase"])

cols_features = [c for c in df.columns if c not in ["Attack_type", "Clase"]]
cols_constantes = [c for c in cols_features if df[c].nunique(dropna=False) <= 1]
if cols_constantes:
    df = df.drop(columns=cols_constantes)
    cols_features = [c for c in cols_features if c not in cols_constantes]

# 4. Eliminar vectores contradictorios (misma X con distinta Clase) y duplicados exactos
#    ANTES de dividir, garantizando 0% de fuga de datos entre Train, Val y Test.
clases_por_x = df.groupby(cols_features)["Clase"].transform("nunique")
n_conflictivos = int((clases_por_x > 1).sum())
df = df[clases_por_x == 1].drop_duplicates(subset=cols_features).reset_index(drop=True)
n_unicos = len(df)

X = df[cols_features].to_numpy(dtype=np.float64)
le = LabelEncoder()
y = le.fit_transform(df["Clase"])

print(f"Registros originales: {n_original:,} | Conflictivos removidos: {n_conflictivos:,} | Únicos limpios: {n_unicos:,}")
print(f"Columnas constantes eliminadas: {cols_constantes}")
print(f"Descriptores numéricos conservados: {len(cols_features)}")
print("Clases codificadas:", dict(zip(le.classes_, le.transform(le.classes_))))

# 5. División aleatoria estratificada en 3 conjuntos DISJUNTOS (70% Train / 15% Val / 15% Test)
#    Se realiza ANTES de escalar y ANTES de balancear para eliminar todo data leakage.
X_train_raw, X_temp_raw, y_train_raw, y_temp_raw = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
    X_temp_raw, y_temp_raw, test_size=0.50, random_state=42, stratify=y_temp_raw
)

# 6. Balanceo de VAL y TEST -> igualar todas las clases a la menor (datos 100% reales y sin copias)
rus_val = RandomUnderSampler(sampling_strategy="all", random_state=42)
X_val_bal, y_val_bal = rus_val.fit_resample(X_val_raw, y_val_raw)

rus_test = RandomUnderSampler(sampling_strategy="all", random_state=42)
X_test_bal, y_test_bal = rus_test.fit_resample(X_test_raw, y_test_raw)

# 7. Balanceo EXCLUSIVO de TRAIN -> 15,000 por clase (60,000 instancias)
#    Usamos SMOTE (en lugar de duplicación exacta) en espacio estandarizado de Train
#    para reducir el sobreajuste por memorización de copias idénticas.
objetivo_train = 15000
conteo_train = pd.Series(y_train_raw).value_counts()

scaler_pre = StandardScaler()
X_train_pre = scaler_pre.fit_transform(X_train_raw)

estrategia_under = {
    clase: objetivo_train
    for clase, cantidad in conteo_train.items()
    if cantidad > objetivo_train
}
if estrategia_under:
    rus_train = RandomUnderSampler(sampling_strategy=estrategia_under, random_state=42)
    X_train_bal_pre, y_train_bal = rus_train.fit_resample(X_train_pre, y_train_raw)
else:
    X_train_bal_pre, y_train_bal = X_train_pre, y_train_raw

estrategia_over = {
    clase: objetivo_train
    for clase, cantidad in pd.Series(y_train_bal).value_counts().items()
    if cantidad < objetivo_train
}
if estrategia_over:
    smote_train = SMOTE(sampling_strategy=estrategia_over, k_neighbors=5, random_state=42)
    X_train_bal_pre, y_train_bal = smote_train.fit_resample(X_train_bal_pre, y_train_bal)

X_train_bal_orig = scaler_pre.inverse_transform(X_train_bal_pre)

# 8. Estandarización final (fit exclusivamente sobre TRAIN balanceado -> Media=0, Std=1)
scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train_bal_orig)
X_val_scaled = scaler_final.transform(X_val_bal)
X_test_scaled = scaler_final.transform(X_test_bal)

# 9. Guardar datos procesados para que los experimentos carguen al instante
np.savez_compressed(
    carpeta_dataset / "datos_procesados.npz",
    X_train=X_train_scaled,
    y_train=y_train_bal,
    X_val=X_val_scaled,
    y_val=y_val_bal,
    X_test=X_test_scaled,
    y_test=y_test_bal,
    clases=le.classes_,
    feature_names=np.array(cols_features),
)

print("\n--- RESUMEN FINAL ---")
print("Distribución TRAIN:", pd.Series(y_train_bal).value_counts().sort_index().to_dict())
print("Distribución VAL:  ", pd.Series(y_val_bal).value_counts().sort_index().to_dict())
print("Distribución TEST: ", pd.Series(y_test_bal).value_counts().sort_index().to_dict())
print(
    "Total instancias procesadas:",
    len(y_train_bal) + len(y_val_bal) + len(y_test_bal),
)
print("Archivo guardado en:", carpeta_dataset / "datos_procesados.npz")
