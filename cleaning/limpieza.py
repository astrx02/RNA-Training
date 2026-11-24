from pathlib import Path
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Cargar dataset
carpeta_dataset = Path(__file__).resolve().parent.parent / "dataset"
archivo_csv = [f for f in carpeta_dataset.rglob("*RT*") if f.is_file()][0]
df = pd.read_csv(archivo_csv)

# 2. Eliminar columnas no numéricas o índices residuales
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

X = df.drop(columns=["Attack_type", "Clase"]).values
le = LabelEncoder()
y = le.fit_transform(df["Clase"])

print("Clases codificadas:", dict(zip(le.classes_, le.transform(le.classes_))))

# 4. División aleatoria estratificada (80% train / 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5. Balanceo de TRAIN -> 25,000 por clase (100,000 instancias)
objetivo_train = 25000
conteo_train = pd.Series(y_train).value_counts()

estrategia_under = {
    clase: objetivo_train
    for clase, cantidad in conteo_train.items()
    if cantidad > objetivo_train
}
if estrategia_under:
    rus_train = RandomUnderSampler(sampling_strategy=estrategia_under, random_state=42)
    X_train_bal, y_train_bal = rus_train.fit_resample(X_train, y_train)
else:
    X_train_bal, y_train_bal = X_train, y_train

estrategia_over = {
    clase: objetivo_train
    for clase, cantidad in pd.Series(y_train_bal).value_counts().items()
    if cantidad < objetivo_train
}
if estrategia_over:
    ros_train = RandomOverSampler(sampling_strategy=estrategia_over, random_state=42)
    X_train_bal, y_train_bal = ros_train.fit_resample(X_train_bal, y_train_bal)

# 6. Balanceo de TEST -> igualar todas las clases a la menor (datos 100% reales)
rus_test = RandomUnderSampler(sampling_strategy="all", random_state=42)
X_test_bal, y_test_bal = rus_test.fit_resample(X_test, y_test)

# 7. Estandarización (fit solo en train para evitar data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test_bal)

# 8. Guardar datos procesados para que main.py los cargue al instante
np.savez_compressed(
    carpeta_dataset / "datos_procesados.npz",
    X_train=X_train_scaled,
    y_train=y_train_bal,
    X_test=X_test_scaled,
    y_test=y_test_bal,
    clases=le.classes_,
)

print("\n--- RESUMEN FINAL ---")
print("Distribución TRAIN:", pd.Series(y_train_bal).value_counts().to_dict())
print("Distribución TEST: ", pd.Series(y_test_bal).value_counts().to_dict())
print("Total instancias listas:", len(y_train_bal) + len(y_test_bal))
print("Archivo guardado en:", carpeta_dataset / "datos_procesados.npz")