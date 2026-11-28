import copy
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
# - h2 < 16 => 1 capa oculta (25% del rango); h2 >= 16 => 2 capas ocultas (75% del rango)
# - alpha en [1e-4, 1.0] para permitir regularización L2 fuerte contra sobreajuste
# - lr en [1e-4, ~0.0316] para evitar inestabilidad/picos de pérdida en Adam
LB = np.array([16.0, 0.0, -4.0, -4.0, 0.0])
UB = np.array([128.0, 64.0, 0.0, -1.5, 2.999999])
ACTIVACIONES = ["relu", "tanh", "logistic"]


def cargar_datos(ruta_npz=None):
    """Carga los arreglos limpios, balanceados, disjuntos y escalados desde dataset/."""
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
        datos["X_val"],
        datos["y_val"],
        datos["X_test"],
        datos["y_test"],
        datos["clases"],
    )


def decodificar_individuo(vector):
    """Convierte el vector continuo de 5 dimensiones en hiperparámetros de MLPClassifier."""
    v = np.clip(vector, LB, UB)
    h1 = int(round(v[0]))
    h2 = int(round(v[1]))

    # Si h2 < 16, la red usa 1 sola capa oculta (modelo más simple contra sobreajuste)
    hidden_layers = (h1,) if h2 < 16 else (h1, h2)
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
        X_val,
        y_val,
        X_test,
        y_test,
        clases,
        eval_sample_size=None,
        max_iter_busqueda=30,
        seed=42,
    ):
        """
        X_val / y_val y X_test / y_test son conjuntos 100% reales y disjuntos de X_train
        (generados antes de SMOTE y antes de StandardScaler), garantizando 0% de data leakage.
        """
        self.X_train_full = X_train
        self.y_train_full = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.clases = clases
        self.max_iter_busqueda = max_iter_busqueda
        self.seed = seed

        # Submuestra estratificada SOLO de X_train para acelerar la búsqueda si se solicita
        if eval_sample_size is not None and eval_sample_size < len(X_train):
            self.X_subtrain, _, self.y_subtrain, _ = train_test_split(
                X_train,
                y_train,
                train_size=eval_sample_size,
                stratify=y_train,
                random_state=seed,
            )
        else:
            self.X_subtrain, self.y_subtrain = X_train, y_train

    def evaluar(self, vector):
        """Entrena un MLPClassifier candidato y devuelve su F1-Score Macro sobre X_val real."""
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

    def entrenar_y_auditar_final(self, mejor_vector, epocas=40, paciencia=8, min_delta=1e-4):
        """
        Entrena el modelo ganador época por época sobre TODO X_train_full y audita sobre X_val real:
        1) Calcula train_loss y val_loss con la misma métrica (log_loss post-época).
        2) Aplica Early Stopping con restauración de mejores pesos (menor val_loss) para evitar sobreajuste.
        3) Evalúa definitivamente el mejor punto de control sobre X_test (datos nunca vistos).
        """
        params = decodificar_individuo(mejor_vector)
        X_tr, y_tr = self.X_train_full, self.y_train_full
        X_va, y_va = self.X_val, self.y_val

        # early_stopping debe ser False al controlar las épocas manualmente con partial_fit
        modelo = MLPClassifier(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            alpha=params["alpha"],
            learning_rate_init=params["learning_rate_init"],
            activation=params["activation"],
            batch_size=512,
            early_stopping=False,
            random_state=self.seed,
        )

        clases_unicas = np.unique(y_tr)
        curvas = {
            "epocas": [],
            "train_f1": [],
            "val_f1": [],
            "train_loss": [],
            "val_loss": [],
            "mejor_epoca": 1,
            "detenido_por_early_stopping": False,
        }

        mejor_modelo = None
        mejor_val_loss = np.inf
        mejor_epoca = 1
        sin_mejora = 0

        rng = np.random.default_rng(self.seed)
        for epoca in range(1, epocas + 1):
            idx = rng.permutation(len(X_tr))
            modelo.partial_fit(X_tr[idx], y_tr[idx], classes=clases_unicas)

            # Evaluación homogénea post-época (misma función de pérdida en Train y Val)
            prob_tr = modelo.predict_proba(X_tr)
            prob_va = modelo.predict_proba(X_va)
            pred_tr = np.argmax(prob_tr, axis=1)
            pred_va = np.argmax(prob_va, axis=1)

            tr_f1 = float(f1_score(y_tr, pred_tr, average="macro"))
            va_f1 = float(f1_score(y_va, pred_va, average="macro"))
            tr_loss = float(log_loss(y_tr, prob_tr, labels=clases_unicas))
            va_loss = float(log_loss(y_va, prob_va, labels=clases_unicas))

            curvas["epocas"].append(epoca)
            curvas["train_f1"].append(tr_f1)
            curvas["val_f1"].append(va_f1)
            curvas["train_loss"].append(tr_loss)
            curvas["val_loss"].append(va_loss)

            # Checkpoint de mejores pesos + Early Stopping anti-sobreajuste
            if va_loss < mejor_val_loss - min_delta:
                mejor_val_loss = va_loss
                mejor_epoca = epoca
                mejor_modelo = copy.deepcopy(modelo)
                sin_mejora = 0
            else:
                sin_mejora += 1
                if sin_mejora >= paciencia:
                    curvas["detenido_por_early_stopping"] = True
                    break

        curvas["mejor_epoca"] = mejor_epoca
        idx_mejor = mejor_epoca - 1
        curvas["brecha_f1_mejor_epoca"] = abs(
            curvas["train_f1"][idx_mejor] - curvas["val_f1"][idx_mejor]
        )

        # Evaluación definitiva con el MEJOR checkpoint sobre el conjunto TEST intacto
        y_pred_test = mejor_modelo.predict(self.X_test)
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
            "modelo": mejor_modelo,
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
