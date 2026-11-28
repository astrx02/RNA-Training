import time
import numpy as np
from optimization.mlp_evaluator import LB, UB, decodificar_individuo


class HibridoGAGWO:
    def __init__(
        self,
        evaluador,
        n_poblacion=12,
        max_iter=15,
        pc=0.75,
        pm=0.15,
        seed=42,
    ):
        self.evaluador = evaluador
        self.n_poblacion = max(4, int(n_poblacion))
        self.max_iter = int(max_iter)
        self.pc = float(pc)
        self.pm = float(pm)
        self.rng = np.random.default_rng(seed)
        self.dim = len(LB)

    def _seleccion_torneo(self, poblacion, fitness, k=3, excluir_idx=None):
        disponibles = [i for i in range(len(poblacion)) if i != excluir_idx]
        k = min(k, len(disponibles))
        indices = self.rng.choice(disponibles, size=k, replace=False)
        mejor_idx = int(indices[np.argmax(fitness[indices])])
        return mejor_idx, poblacion[mejor_idx].copy()

    @staticmethod
    def _clave_vector(vec):
        return tuple(np.round(vec, 6))

    def _actualizar_lideres_distintos(self, poblacion, fitness, lideres_previos):
        """
        Selecciona los 3 mejores individuos DISTINTOS (Alfa, Beta, Delta) evitando
        el colapso de los 3 líderes a clones idénticos de Alfa.
        """
        unicos = {}
        for vec, fit in zip(poblacion, fitness):
            clave = self._clave_vector(vec)
            fit_f = float(fit)
            if clave not in unicos or fit_f > unicos[clave][1]:
                unicos[clave] = (vec.copy(), fit_f)

        for vec, fit in lideres_previos:
            if fit > -np.inf:
                clave = self._clave_vector(vec)
                fit_f = float(fit)
                if clave not in unicos or fit_f > unicos[clave][1]:
                    unicos[clave] = (vec.copy(), fit_f)

        ordenados = sorted(unicos.values(), key=lambda par: par[1], reverse=True)
        while len(ordenados) < 3:
            ordenados.append((ordenados[-1][0].copy(), ordenados[-1][1]))

        return ordenados[0], ordenados[1], ordenados[2]

    def ejecutar(self, callback=None):
        """
        Ejecuta la optimización híbrida GA + GWO.
        callback(stats_iteracion): función opcional que la interfaz puede pasar
                                   para actualizar barras de progreso o logs en vivo.
        """
        inicio = time.time()

        # 1. Población inicial uniforme en [LB, UB] (Base GA)
        poblacion = self.rng.uniform(LB, UB, size=(self.n_poblacion, self.dim))

        # Líderes globales distintos de GWO: Alfa, Beta, Delta
        x_alfa, fit_alfa = np.zeros(self.dim), -np.inf
        x_beta, fit_beta = np.zeros(self.dim), -np.inf
        x_delta, fit_delta = np.zeros(self.dim), -np.inf

        historial = []
        cache_fitness = {}  # Evita re-evaluar los 3 líderes elitistas

        # 2. Ciclo evolutivo
        for t in range(1, self.max_iter + 1):
            fitness = np.zeros(self.n_poblacion)

            # --- FASE A: Evaluación ---
            for i in range(self.n_poblacion):
                clave = self._clave_vector(poblacion[i])
                if clave in cache_fitness:
                    fitness[i] = cache_fitness[clave]
                else:
                    score = self.evaluador.evaluar(poblacion[i])
                    fitness[i] = score
                    cache_fitness[clave] = score

            # --- FASE B: Actualizar jerarquía global de 3 líderes DISTINTOS (GWO) ---
            (x_alfa, fit_alfa), (x_beta, fit_beta), (x_delta, fit_delta) = (
                self._actualizar_lideres_distintos(
                    poblacion,
                    fitness,
                    [(x_alfa, fit_alfa), (x_beta, fit_beta), (x_delta, fit_delta)],
                )
            )

            # Factor de control lineal a en [2.0, 0.0] sobre las iteraciones reales
            a = 2.0 * (1.0 - (t - 1) / max(1, self.max_iter - 1))

            # --- Registro estadístico para el Punto 4 ---
            stats_iter = {
                "iteracion": t,
                "n_poblacion": self.n_poblacion,
                "parametro_a": round(float(a), 4),
                "fit_alfa": fit_alfa,
                "fit_beta": fit_beta,
                "fit_delta": fit_delta,
                "media": float(np.mean(fitness)),
                "mediana": float(np.median(fitness)),
                "desv_std": float(np.std(fitness)),
                "min_fitness": float(np.min(fitness)),
                "max_fitness": float(np.max(fitness)),
                "fitness_poblacion": [float(f) for f in fitness],
                "mejores_hiperparametros": decodificar_individuo(x_alfa),
            }
            historial.append(stats_iter)

            if callback is not None:
                callback(stats_iter)

            # En la última iteración ya no se desperdicia cómputo creando una generación no evaluada
            if t == self.max_iter:
                break

            # --- FASE C: Nueva generación (Base GA + Guía GWO) ---
            # Elitismo con 3 líderes distintos (nunca clones de Alfa)
            nueva_poblacion = [x_alfa.copy(), x_beta.copy(), x_delta.copy()]
            claves_en_nueva = {self._clave_vector(v) for v in nueva_poblacion}

            while len(nueva_poblacion) < self.n_poblacion:
                # 1. Selección por torneo de 2 padres distintos (GA)
                idx1, padre1 = self._seleccion_torneo(poblacion, fitness, k=3)
                _, padre2 = self._seleccion_torneo(
                    poblacion, fitness, k=3, excluir_idx=idx1
                )

                # 2. Cruce aritmético (GA)
                if self.rng.random() < self.pc:
                    w = self.rng.random()
                    hijo = w * padre1 + (1.0 - w) * padre2
                else:
                    hijo = padre1.copy()

                # 3. Guía de caza hacia Alfa, Beta y Delta distintos (GWO)
                r1, r2 = self.rng.random(self.dim), self.rng.random(self.dim)
                A1, C1 = 2.0 * a * r1 - a, 2.0 * r2
                D_alfa = np.abs(C1 * x_alfa - hijo)
                X1 = x_alfa - A1 * D_alfa

                r3, r4 = self.rng.random(self.dim), self.rng.random(self.dim)
                A2, C2 = 2.0 * a * r3 - a, 2.0 * r4
                D_beta = np.abs(C2 * x_beta - hijo)
                X2 = x_beta - A2 * D_beta

                r5, r6 = self.rng.random(self.dim), self.rng.random(self.dim)
                A3, C3 = 2.0 * a * r5 - a, 2.0 * r6
                D_delta = np.abs(C3 * x_delta - hijo)
                X3 = x_delta - A3 * D_delta

                hijo = (X1 + X2 + X3) / 3.0

                # 4. Mutación gaussiana (GA)
                mascara_mut = self.rng.random(self.dim) < self.pm
                if np.any(mascara_mut):
                    ruido = self.rng.normal(0.0, 0.1 * (UB - LB))
                    hijo[mascara_mut] += ruido[mascara_mut]

                # 5. Acotar a los límites permitidos y evitar clones exactos
                hijo = np.clip(hijo, LB, UB)
                if self._clave_vector(hijo) in claves_en_nueva:
                    hijo = np.clip(
                        hijo + self.rng.normal(0.0, 0.02 * (UB - LB)), LB, UB
                    )

                claves_en_nueva.add(self._clave_vector(hijo))
                nueva_poblacion.append(hijo)

            poblacion = np.array(nueva_poblacion)

        duracion = round(time.time() - inicio, 2)
        return {
            "mejor_vector": x_alfa.tolist(),
            "mejores_hiperparametros": decodificar_individuo(x_alfa),
            "mejor_fitness_val": fit_alfa,
            "tiempo_segundos": duracion,
            "historial": historial,
        }
