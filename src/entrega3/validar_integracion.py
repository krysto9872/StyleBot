"""Comprueba que los componentes de la Entrega 3 usan el protocolo común.

Responsable: Krysto Alberto Adan Jauregui Zúñiga (líder y responsable de datos).

Uso desde la raíz del repositorio:
    python src/entrega3/validar_integracion.py
    python src/entrega3/validar_integracion.py --estricto

La ejecución normal valida el protocolo y avisa qué entregables siguen pendientes.
Con ``--estricto`` también devuelve error mientras falte algún archivo final.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gymnasium.utils.env_checker import check_env


RAIZ = Path(__file__).resolve().parents[2]
CARPETA_SRC = Path(__file__).resolve().parent
if str(CARPETA_SRC) not in sys.path:
    sys.path.insert(0, str(CARPETA_SRC))

from entorno_almacen import (  # noqa: E402
    CATEGORIAS,
    ESTANTES,
    NOMBRES_ACCIONES,
    RECOMPENSAS,
    AlmacenStyleBot,
)


ENTREGABLES_FINALES = [
    "src/entrega3/entorno_almacen.py",
    "src/entrega3/agentes_rl.py",
    "src/entrega3/percepcion_cnn.py",
    "notebooks/entrega3/01_qlearning_dynaq.ipynb",
    "notebooks/entrega3/02_integracion_cnn_rl.ipynb",
    "notebooks/entrega3/03_evaluacion_rl.ipynb",
    "notebooks/entrega3/E3_StyleBot.ipynb",
    "modelos/entrega3/q_table_qlearning.npz",
    "modelos/entrega3/q_table_dynaq.npz",
]


def cargar_protocolo() -> dict:
    ruta = RAIZ / "datos" / "entrega3" / "protocolo_entrega3.json"
    with ruta.open(encoding="utf-8") as archivo:
        return json.load(archivo)


def registrar(condicion: bool, mensaje: str, errores: list[str]) -> None:
    marca = "OK" if condicion else "ERROR"
    print(f"[{marca}] {mensaje}")
    if not condicion:
        errores.append(mensaje)


def validar_protocolo() -> list[str]:
    protocolo = cargar_protocolo()
    entorno = AlmacenStyleBot()
    errores: list[str] = []

    print("\n1. Compatibilidad del entorno con el protocolo común")
    registrar(list(entorno.mapa.shape) == protocolo["tamano_mapa"], "Mapa de 8 x 8", errores)
    registrar([list(p) for p in entorno.recepcion] == protocolo["recepcion"], "Zona de recepción", errores)
    registrar(CATEGORIAS == protocolo["categorias"], "Orden de las diez categorías", errores)

    acciones = {str(i): nombre for i, nombre in enumerate(NOMBRES_ACCIONES)}
    registrar(acciones == protocolo["acciones"], "Índices y nombres de las seis acciones", errores)
    registrar(entorno.action_space.n == protocolo["numero_acciones"], "Cantidad de acciones", errores)
    registrar(entorno.observation_space.n == protocolo["numero_estados"], "Cantidad de estados", errores)

    recompensas = {clave: float(valor) for clave, valor in protocolo["recompensas"].items()}
    registrar(RECOMPENSAS == recompensas, "Valores de las recompensas", errores)

    estantes = {str(categoria): list(posicion) for posicion, categoria in ESTANTES.items()}
    registrar(estantes == protocolo["estantes"], "Categoría y posición de cada estante", errores)
    registrar(
        entorno.max_pasos == protocolo["max_pasos_por_prenda"] * entorno.prendas_por_episodio,
        "Límite de 60 pasos por prenda",
        errores,
    )

    print("\n2. Validez técnica del entorno")
    try:
        check_env(entorno, skip_render_check=True)
        registrar(True, "Gymnasium check_env", errores)
    except Exception as exc:  # pragma: no cover - deja el error visible al equipo
        registrar(False, f"Gymnasium check_env: {exc}", errores)

    ida_vuelta = all(
        entorno.codificar(*entorno.decodificar(estado)) == estado
        for estado in range(entorno.observation_space.n)
    )
    registrar(ida_vuelta, "Codificación reversible de los 1,408 estados", errores)

    alcanzables = entorno.distancias_desde(entorno.inicio)
    registrar(all(posicion in alcanzables for posicion in ESTANTES), "Todos los estantes son alcanzables", errores)

    return errores


def revisar_entregables() -> list[str]:
    print("\n3. Estado de los archivos finales")
    faltantes: list[str] = []
    for ruta_relativa in ENTREGABLES_FINALES:
        existe = (RAIZ / ruta_relativa).is_file()
        print(f"[{'OK' if existe else 'PENDIENTE'}] {ruta_relativa}")
        if not existe:
            faltantes.append(ruta_relativa)
    return faltantes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--estricto",
        action="store_true",
        help="también marca error si faltan entregables finales",
    )
    argumentos = parser.parse_args()

    errores = validar_protocolo()
    faltantes = revisar_entregables()

    print("\nResumen")
    if errores:
        print(f"La integración común tiene {len(errores)} error(es) que deben corregirse.")
    else:
        print("El entorno actual es compatible con el protocolo común de la Entrega 3.")
    print(f"Archivos finales pendientes: {len(faltantes)} de {len(ENTREGABLES_FINALES)}.")

    if errores or (argumentos.estricto and faltantes):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
