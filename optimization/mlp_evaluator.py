from pathlib import Path
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Límites del espacio de búsqueda: [h1, h2, log10(alpha), log10(lr), activacion]
LB = np.array([16.0, 0.0, -5.0, -4.0, 0.0])
UB = np.array([128.0, 64.0, -1.0, -1.0, 2.99])
ACTIVACIONES = ["relu", "tanh", "logistic"]


def cargar_datos(ruta_npz=None):
    """Carga los arreglos limpios, balanceados y escalados desde dataset/."""
    if ruta_npz is None:
        ruta_npz = (
            Path(__file__).resolve().parent.parent
            / "dataset"
            / "datos_procesados.npz"
        )
    datos = np.load(ruta_npz, allow_pickle=True)
    return (
        datos["X_train"],
        datos["y_train"],
        datos["X_test"],
        datos["y_test"],
        datos["clases"],
    )


def decodificar_individuo(vector):
    """Convierte el vector continuo de 5 dimensiones en hiperparámetros de MLPClassifier."""
    v = np.clip(vector, LB, UB)
    h1 = int(round(v[0]))
    h2 = int(round(v[1]))

    # Si h2 es menor a 4, la red usa 1 sola capa oculta; si no, usa 2 capas
    hidden_layers = (h1,) if h2 < 4 else (h1, h2)
    alpha = float(10.0 ** v[2])
    lr = float(10.0 ** v[3])
    act_idx = int(np.clip(np.floor(v[4]), 0, len(ACTIVACIONES) - 1))
    activation = ACTIVACIONES[act_idx]

    return {
        "hidden_layer_sizes": hidden_layers,
        "alpha": alpha,
        "learning_rate_init": lr,
        "activation": activation,
    }


class MLPEvaluator:
    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        clases,
        val_size=0.15,
        eval_sample_size=None,
        max_iter_busqueda=30,
        seed=42,
    ):
        """
        eval_sample_size: si es None usa los 100,000 datos en la búsqueda.
                          Puedes pasarlo en ej. 20000 para pruebas rápidas en UI;
                          el modelo final SIEMPRE se entrena con todo X_train.
        """
        self.X_train_full = X_train
        self.y_train_full = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.clases = clases
        self.max_iter_busqueda = max_iter_busqueda
        self.seed = seed

        # Subdivisión estratificada interna para evaluar fitness sin tocar X_test
        X_base, y_base = X_train, y_train
        if eval_sample_size is not None and eval_sample_size < len(X_train):
            X_base, _, y_base, _ = train_test_split(
                X_train,
                y_train,
                train_size=eval_sample_size,
                stratify=y_train,
                random_state=seed,
            )

        self.X_subtrain, self.X_val, self.y_subtrain, self.y_val = (
            train_test_split(
                X_base,
                y_base,
                test_size=val_size,
                stratify=y_base,
                random_state=seed,
            )
        )

    def evaluar(self, vector):
        """Entrena un MLPClassifier candidato y devuelve su F1-Score Macro en validación."""
        params = decodificar_individuo(vector)
        modelo = MLPClassifier(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            alpha=params["alpha"],
            learning_rate_init=params["learning_rate_init"],
            activation=params["activation"],
            batch_size=512,
            max_iter=self.max_iter_busqueda,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=self.seed,
        )
        modelo.fit(self.X_subtrain, self.y_subtrain)
        y_pred = modelo.predict(self.X_val)
        return float(f1_score(self.y_val, y_pred, average="macro"))

    def entrenar_y_auditar_final(self, mejor_vector, epocas=40):
        """
        Entrena el modelo ganador época por época sobre el dataset completo para:
        1) Obtener curvas Train vs. Validación (demostrar ausencia de sobreajuste).
        2) Evaluar el desempeño final sobre X_test (datos nunca vistos).
        """
        params = decodificar_individuo(mejor_vector)
        X_tr, X_va, y_tr, y_va = train_test_split(
            self.X_train_full,
            self.y_train_full,
            test_size=0.15,
            stratify=self.y_train_full,
            random_state=self.seed,
        )

        # early_stopping debe ser False al usar partial_fit época por época
        modelo = MLPClassifier(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            alpha=params["alpha"],
            learning_rate_init=params["learning_rate_init"],
            activation=params["activation"],
            batch_size=512,
            early_stopping=False,
            random_state=self.seed,
        )

        clases_unicas = np.unique(self.y_train_full)
        curvas = {
            "epocas": [],
            "train_f1": [],
            "val_f1": [],
            "train_loss": [],
            "val_loss": [],
        }

        rng = np.random.default_rng(self.seed)
        for epoca in range(1, epocas + 1):
            # Barajar en cada época para estabilizar el gradiente
            idx = rng.permutation(len(X_tr))
            modelo.partial_fit(X_tr[idx], y_tr[idx], classes=clases_unicas)

            pred_tr = modelo.predict(X_tr)
            pred_va = modelo.predict(X_va)
            prob_va = modelo.predict_proba(X_va)

            curvas["epocas"].append(epoca)
            curvas["train_f1"].append(
                float(f1_score(y_tr, pred_tr, average="macro"))
            )
            curvas["val_f1"].append(
                float(f1_score(y_va, pred_va, average="macro"))
            )
            curvas["train_loss"].append(float(modelo.loss_))
            curvas["val_loss"].append(float(log_loss(y_va, prob_va)))

        # Evaluación definitiva en el conjunto TEST intacto
        y_pred_test = modelo.predict(self.X_test)
        test_f1 = float(f1_score(self.y_test, y_pred_test, average="macro"))
        test_acc = float(accuracy_score(self.y_test, y_pred_test))
        matriz_conf = confusion_matrix(self.y_test, y_pred_test).tolist()
        reporte = classification_report(
            self.y_test,
            y_pred_test,
            target_names=[str(c) for c in self.clases],
            output_dict=True,
        )

        return {
            "modelo": modelo,
            "hiperparametros": params,
            "curvas_entrenamiento": curvas,
            "metricas_test": {
                "f1_macro": test_f1,
                "accuracy": test_acc,
                "matriz_confusion": matriz_conf,
                "reporte_clases": reporte,
                "clases": [str(c) for c in self.clases],
            },
        }