"""Entorno Gymnasium del almacén de StyleBot (Entrega 3).

Responsable: Luis Angel Mendoza Gaytán (simulación y aprendizaje por refuerzo).

El robot recoge prendas en la zona de recepción y las lleva al estante de su
categoría. La categoría que "ve" el robot es la predicha por la CNN; la
recompensa se calcula con la categoría real, como pasaría en un almacén real.

Mapa de 8 x 8 (fila 0 arriba):

    0 1 2 3 4 5 6 7
  0 E E E E E E E E     E = estante (categorías 0 a 7)
  1 . . . . . . . .     . = pasillo
  2 . # # . . # # .     # = obstáculo
  3 . . . . . . . .
  4 E . # . . # . E     estante 8 (Bolso) a la izquierda, 9 (Botín) a la derecha
  5 . . . . . . . .
  6 . # . . . . # .
  7 . . . R R . . .     R = recepción

MDP
---
Estado (discreto, para Q-Learning y Dyna-Q tabulares):
    posición del robot (64) x prenda cargada (11: ninguna o categoría 0-9 percibida)
    x espacio en el estante de esa prenda (2: hay espacio / lleno) = 1,408 estados.
    De la ocupación de los estantes solo entra al estado lo que cambia la decisión:
    si el estante de la prenda cargada tiene espacio. La ocupación completa va en `info`.

Acciones (6): 0 arriba, 1 abajo, 2 izquierda, 3 derecha, 4 tomar, 5 colocar.
    - tomar: solo en recepción, sin carga y con prendas pendientes.
    - colocar: sobre un estante deja la prenda. Sobre recepción la devuelve a revisión
      manual, pero solo si el estante de esa prenda está lleno (si no, es inválida).

Recompensas (las sugeridas por el proyecto, ajustables con `recompensas=`):
    +10 prenda en el estante correcto
     -8 prenda en un estante incorrecto
     -3 intento de colocar en un estante lleno, o prenda devuelta porque su estante está lleno
     -2 acción inválida (chocar con pared u obstáculo, tomar o colocar donde no se puede)
     -1 cualquier otro paso
    Cada acción recibe una sola de estas recompensas; el costo por paso ya está incluido.
    La acción inválida cuesta -2 para que el robot aprenda a no repetirla, en lugar de
    confundirla con un paso normal.

Terminación: todas las prendas del episodio fueron acomodadas o devueltas.
Truncamiento: se alcanzó `max_pasos`.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

CATEGORIAS = [
    "Camiseta/top", "Pantalón", "Suéter", "Vestido", "Abrigo",
    "Sandalia", "Camisa", "Tenis", "Bolso", "Botín",
]

MAPA = [
    "EEEEEEEE",
    "........",
    ".##..##.",
    "........",
    "E.#..#.E",
    "........",
    ".#....#.",
    "...RR...",
]

# Categoría de cada estante (fila, columna) -> índice de CATEGORIAS
ESTANTES = {(0, c): c for c in range(8)}
ESTANTES[(4, 0)] = 8
ESTANTES[(4, 7)] = 9

ARRIBA, ABAJO, IZQUIERDA, DERECHA, TOMAR, COLOCAR = range(6)
NOMBRES_ACCIONES = ["arriba", "abajo", "izquierda", "derecha", "tomar", "colocar"]
MOVIMIENTOS = {ARRIBA: (-1, 0), ABAJO: (1, 0), IZQUIERDA: (0, -1), DERECHA: (0, 1)}

RECOMPENSAS = {
    "correcto": 10.0,
    "incorrecto": -8.0,
    "lleno": -3.0,
    "devuelta": -3.0,
    "invalida": -2.0,
    "paso": -1.0,
}

SIN_CARGA = 10  # valor de "carga" cuando el robot no lleva prenda


class AlmacenStyleBot(gym.Env):
    """Almacén de 8 x 8 con recepción, pasillos, obstáculos y 10 estantes por categoría.

    Parámetros
    ----------
    etiquetas : array de categorías reales (0-9) de las prendas disponibles.
    predicciones : array de categorías predichas por la CNN, mismo tamaño que `etiquetas`.
        Si no se da, la percepción es perfecta (predicción = etiqueta real).
    prendas_por_episodio : prendas que llegan a recepción en cada episodio.
    capacidad : prendas que caben en cada estante.
    prob_estante_lleno : probabilidad de que un estante empiece lleno en `reset`; sirve
        para que el agente aprenda qué hacer con un estante lleno.
    max_pasos : límite de pasos por episodio (por defecto 60 por prenda).
    recompensas : diccionario para cambiar cualquiera de los valores de RECOMPENSAS.
    render_mode : None, "ansi" (texto) o "rgb_array" (imagen para el GIF).
    """

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 6}

    def __init__(self, etiquetas=None, predicciones=None, prendas_por_episodio=5, capacidad=8,
                 prob_estante_lleno=0.0, max_pasos=None, recompensas=None, render_mode=None):
        super().__init__()
        self.mapa = np.array([list(fila) for fila in MAPA])
        self.alto, self.ancho = self.mapa.shape
        self.recepcion = [tuple(p) for p in np.argwhere(self.mapa == "R")]
        self.inicio = self.recepcion[0]

        if etiquetas is None:
            etiquetas = np.arange(10)
        self.etiquetas = np.asarray(etiquetas, dtype=np.int64)
        self.predicciones = self.etiquetas.copy() if predicciones is None else np.asarray(predicciones, dtype=np.int64)
        if self.predicciones.shape != self.etiquetas.shape:
            raise ValueError("etiquetas y predicciones deben tener el mismo tamaño")
        if not (np.all((0 <= self.etiquetas) & (self.etiquetas < 10)) and np.all((0 <= self.predicciones) & (self.predicciones < 10))):
            raise ValueError("las categorías deben estar entre 0 y 9")

        self.prendas_por_episodio = int(prendas_por_episodio)
        self.capacidad = int(capacidad)
        self.prob_estante_lleno = float(prob_estante_lleno)
        self.max_pasos = int(max_pasos) if max_pasos else 60 * self.prendas_por_episodio
        self.recompensas = {**RECOMPENSAS, **(recompensas or {})}
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode no válido: {render_mode}")
        self.render_mode = render_mode

        self.n_estados = self.alto * self.ancho * 11 * 2
        self.observation_space = spaces.Discrete(self.n_estados)
        self.action_space = spaces.Discrete(6)
        self.posicion_estante = {categoria: pos for pos, categoria in ESTANTES.items()}
        self._orden_fijo = None

    # ------------------------------------------------------------------ estado
    def codificar(self, fila, columna, carga, destino_lleno):
        return ((fila * self.ancho + columna) * 11 + carga) * 2 + int(destino_lleno)

    def decodificar(self, estado):
        estado = int(estado)
        destino_lleno = estado % 2
        estado //= 2
        carga = estado % 11
        celda = estado // 11
        return celda // self.ancho, celda % self.ancho, carga, bool(destino_lleno)

    def _destino_lleno(self):
        if self.carga_percibida == SIN_CARGA:
            return False
        return self.ocupacion[self.carga_percibida] >= self.capacidad

    def _observacion(self):
        fila, columna = self.posicion
        return np.int64(self.codificar(fila, columna, self.carga_percibida, self._destino_lleno()))

    def _info(self, evento="inicio"):
        return {
            "evento": evento,
            "posicion": self.posicion,
            "carga_percibida": None if self.carga_percibida == SIN_CARGA else int(self.carga_percibida),
            "carga_real": None if self.carga_real is None else int(self.carga_real),
            "prendas_pendientes": len(self.cola),
            "ocupacion": self.ocupacion.copy(),
            "pasos": self.pasos,
            **{clave: valor for clave, valor in self.contadores.items()},
        }

    # --------------------------------------------------------------- dinámica
    def reset(self, *, seed=None, options=None):
        """Reinicia el almacén.

        options:
            "indices": lista de índices de `etiquetas` que llegarán en ese orden
                       (por ejemplo, las 50 prendas de la simulación final).
            "ocupacion": lista de 10 enteros con la ocupación inicial de los estantes.
        """
        super().reset(seed=seed)
        options = options or {}
        if "indices" in options:
            indices = np.asarray(options["indices"], dtype=np.int64)
        else:
            indices = self.np_random.integers(0, len(self.etiquetas), size=self.prendas_por_episodio)
        self.cola = list(indices)
        self.total_prendas = len(self.cola)

        if "ocupacion" in options:
            self.ocupacion = np.asarray(options["ocupacion"], dtype=np.int64).copy()
        else:
            llenos = self.np_random.random(10) < self.prob_estante_lleno
            self.ocupacion = np.where(llenos, self.capacidad, 0).astype(np.int64)

        self.posicion = self.inicio
        self.carga_percibida = SIN_CARGA
        self.carga_real = None
        self.indice_cargado = None
        self.pasos = 0
        self.contadores = {
            "correctas": 0, "incorrectas": 0, "fallos_cnn": 0, "fallos_politica": 0,
            "intentos_estante_lleno": 0, "devueltas": 0, "acciones_invalidas": 0,
        }
        self.historial_colocadas = []  # (índice, real, percibida, estante)
        return self._observacion(), self._info()

    def step(self, accion):
        if not self.action_space.contains(accion):
            raise ValueError(f"Acción no válida: {accion}")
        accion = int(accion)
        self.pasos += 1
        r = self.recompensas
        recompensa, evento = r["paso"], NOMBRES_ACCIONES[accion]

        if accion in MOVIMIENTOS:
            df, dc = MOVIMIENTOS[accion]
            fila, columna = self.posicion[0] + df, self.posicion[1] + dc
            if 0 <= fila < self.alto and 0 <= columna < self.ancho and self.mapa[fila, columna] != "#":
                self.posicion = (fila, columna)
            else:
                recompensa, evento = r["invalida"], "choque"
                self.contadores["acciones_invalidas"] += 1

        elif accion == TOMAR:
            if self.posicion in self.recepcion and self.carga_percibida == SIN_CARGA and self.cola:
                self.indice_cargado = int(self.cola.pop(0))
                self.carga_real = int(self.etiquetas[self.indice_cargado])
                self.carga_percibida = int(self.predicciones[self.indice_cargado])  # percepción de la CNN
                evento = "tomar"
            else:
                recompensa, evento = r["invalida"], "tomar_invalido"
                self.contadores["acciones_invalidas"] += 1

        else:  # COLOCAR
            if self.carga_percibida == SIN_CARGA:
                recompensa, evento = r["invalida"], "colocar_sin_carga"
                self.contadores["acciones_invalidas"] += 1
            elif self.posicion in ESTANTES:
                estante = ESTANTES[self.posicion]
                if self.ocupacion[estante] >= self.capacidad:
                    recompensa, evento = r["lleno"], "estante_lleno"
                    self.contadores["intentos_estante_lleno"] += 1
                else:
                    self.ocupacion[estante] += 1
                    self.historial_colocadas.append((self.indice_cargado, self.carga_real, self.carga_percibida, estante))
                    if estante == self.carga_real:
                        recompensa, evento = r["correcto"], "correcto"
                        self.contadores["correctas"] += 1
                    else:
                        recompensa, evento = r["incorrecto"], "incorrecto"
                        self.contadores["incorrectas"] += 1
                        # Si el robot fue al estante que predijo la CNN, el error es de percepción.
                        clave = "fallos_cnn" if estante == self.carga_percibida else "fallos_politica"
                        self.contadores[clave] += 1
                    self._soltar()
            elif self.posicion in self.recepcion and self._destino_lleno():
                # Solo se puede devolver si el estante de la prenda está lleno; si no, el robot
                # podría aprender a "deshacerse" de prendas en recepción en vez de acomodarlas.
                recompensa, evento = r["devuelta"], "devuelta"
                self.contadores["devueltas"] += 1
                self._soltar()
            else:
                recompensa, evento = r["invalida"], "colocar_invalido"
                self.contadores["acciones_invalidas"] += 1

        terminado = not self.cola and self.carga_percibida == SIN_CARGA
        truncado = (not terminado) and self.pasos >= self.max_pasos
        return self._observacion(), float(recompensa), terminado, truncado, self._info(evento)

    def _soltar(self):
        self.carga_percibida = SIN_CARGA
        self.carga_real = None
        self.indice_cargado = None

    # --------------------------------------------------------------- utilidades
    def celdas_libres(self):
        """Celdas por las que puede pasar el robot (todo excepto obstáculos)."""
        return [(f, c) for f in range(self.alto) for c in range(self.ancho) if self.mapa[f, c] != "#"]

    def distancias_desde(self, origen):
        """Distancia mínima en pasos desde `origen` a cada celda (búsqueda en anchura)."""
        distancias = {origen: 0}
        frontera = [origen]
        while frontera:
            siguiente = []
            for f, c in frontera:
                for df, dc in MOVIMIENTOS.values():
                    vecino = (f + df, c + dc)
                    if (0 <= vecino[0] < self.alto and 0 <= vecino[1] < self.ancho
                            and self.mapa[vecino] != "#" and vecino not in distancias):
                        distancias[vecino] = distancias[(f, c)] + 1
                        siguiente.append(vecino)
            frontera = siguiente
        return distancias

    # ------------------------------------------------------------------ render
    def render(self):
        if self.render_mode == "ansi":
            return self._render_texto()
        if self.render_mode == "rgb_array":
            return self._render_imagen()
        return None

    def _render_texto(self):
        filas = []
        for f in range(self.alto):
            celdas = []
            for c in range(self.ancho):
                if (f, c) == self.posicion:
                    celdas.append("@" if self.carga_percibida == SIN_CARGA else "&")
                elif (f, c) in ESTANTES:
                    celdas.append(str(ESTANTES[(f, c)]))
                else:
                    celdas.append(self.mapa[f, c])
            filas.append(" ".join(celdas))
        carga = "nada" if self.carga_percibida == SIN_CARGA else CATEGORIAS[self.carga_percibida]
        filas.append(f"Carga: {carga} | Pendientes: {len(self.cola)} | Correctas: {self.contadores['correctas']}"
                     f"/{self.total_prendas} | Pasos: {self.pasos}")
        return "\n".join(filas)

    def _render_imagen(self, tam=48):
        colores = {".": (235, 235, 235), "#": (70, 70, 70), "R": (255, 214, 120), "E": (150, 190, 230)}
        imagen = np.zeros((self.alto * tam, self.ancho * tam, 3), dtype=np.uint8)
        for f in range(self.alto):
            for c in range(self.ancho):
                bloque = imagen[f * tam:(f + 1) * tam, c * tam:(c + 1) * tam]
                bloque[:] = colores[self.mapa[f, c]]
                bloque[:1, :] = bloque[:, :1] = 200
                if (f, c) in ESTANTES:  # barra de llenado del estante
                    llenado = int((tam - 8) * min(self.ocupacion[ESTANTES[(f, c)]], self.capacidad) / self.capacidad)
                    bloque[tam - 4 - llenado:tam - 4, tam - 10:tam - 4] = (40, 90, 160)
        f, c = self.posicion
        m = tam // 4
        color_robot = (40, 160, 70) if self.carga_percibida == SIN_CARGA else (210, 60, 50)
        imagen[f * tam + m:(f + 1) * tam - m, c * tam + m:(c + 1) * tam - m] = color_robot
        return imagen


def crear_entorno(**kwargs):
    """Atajo para crear el entorno con los mismos parámetros que AlmacenStyleBot."""
    return AlmacenStyleBot(**kwargs)


if "StyleBot/Almacen-v0" not in gym.registry:
    gym.register(id="StyleBot/Almacen-v0", entry_point=AlmacenStyleBot)


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    entorno = AlmacenStyleBot(render_mode="ansi")
    check_env(entorno, skip_render_check=True)
    obs, info = entorno.reset(seed=42)
    print(entorno.render())
    print("Estados:", entorno.observation_space.n, "| Acciones:", entorno.action_space.n)
