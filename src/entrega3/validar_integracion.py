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
import ast
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


def registrar_aviso(condicion: bool, mensaje: str, avisos: list[str]) -> None:
    """Registra una comprobación que solo será obligatoria en modo estricto."""
    marca = "OK" if condicion else "AVISO"
    print(f"[{marca}] {mensaje}")
    if not condicion:
        avisos.append(mensaje)


def cargar_notebook(ruta_relativa: str) -> dict:
    """Carga un notebook desde el repositorio sin ejecutarlo ni importar TensorFlow."""
    with (RAIZ / ruta_relativa).open(encoding="utf-8") as archivo:
        return json.load(archivo)


def resumir_notebook(notebook: dict) -> dict:
    """Devuelve ejecución, errores, código y salidas guardadas de un notebook."""
    celdas_codigo = [celda for celda in notebook.get("cells", []) if celda.get("cell_type") == "code"]
    salidas = [salida for celda in celdas_codigo for salida in celda.get("outputs", [])]
    errores = [salida for salida in salidas if salida.get("output_type") == "error"]

    def texto_salida(salida: dict) -> str:
        texto = salida.get("text", "")
        if isinstance(texto, list):
            return "".join(texto)
        return str(texto)

    return {
        "codigo": "\n".join("".join(celda.get("source", [])) for celda in celdas_codigo),
        "todo": "\n".join("".join(celda.get("source", [])) for celda in notebook.get("cells", [])),
        "salidas": "\n".join(texto_salida(salida) for salida in salidas),
        "celdas_codigo": len(celdas_codigo),
        "celdas_ejecutadas": sum(celda.get("execution_count") is not None for celda in celdas_codigo),
        "errores": errores,
    }


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


def validar_componentes_nuevos() -> tuple[list[str], list[str]]:
    """Comprueba percepción CNN, integración y evaluación contra el protocolo común."""
    errores: list[str] = []
    avisos: list[str] = []
    print("\n4. Compatibilidad de percepción, integración y evaluación")

    ruta_percepcion = RAIZ / "src" / "entrega3" / "percepcion_cnn.py"
    if ruta_percepcion.is_file():
        fuente = ruta_percepcion.read_text(encoding="utf-8")
        try:
            ast.parse(fuente)
            registrar(True, "percepcion_cnn.py tiene sintaxis válida", errores)
        except SyntaxError as exc:
            registrar(False, f"percepcion_cnn.py tiene sintaxis válida: {exc}", errores)
        registrar("datos/entrega3/protocolo_entrega3.json" in fuente, "Percepción usa el protocolo común", errores)
        registrar("modelos/stylebot_cnn_final.keras" in fuente, "Percepción usa la CNN final establecida", errores)
        registrar(".fit(" not in fuente, "La CNN no se vuelve a entrenar", errores)
        registrar("modelo.trainable = False" in fuente, "La CNN se congela para inferencia", errores)
    else:
        registrar_aviso(False, "Falta src/entrega3/percepcion_cnn.py", avisos)

    ruta_integracion = "notebooks/entrega3/02_integracion_cnn_rl.ipynb"
    if (RAIZ / ruta_integracion).is_file():
        resumen = resumir_notebook(cargar_notebook(ruta_integracion))
        registrar(resumen["celdas_codigo"] > 0, "Integración CNN-RL contiene código", errores)
        registrar(
            resumen["celdas_ejecutadas"] == resumen["celdas_codigo"],
            "Todas las celdas de integración CNN-RL están ejecutadas",
            errores,
        )
        registrar(not resumen["errores"], "Integración CNN-RL no contiene errores guardados", errores)
        registrar("AlmacenStyleBot" in resumen["codigo"], "Integración usa el entorno común", errores)
        registrar("49/50" in resumen["salidas"], "Simulación de 50 prendas conserva su resultado", errores)
        registrar(
            "Fallos de CNN" in resumen["salidas"] and "Fallos política" in resumen["salidas"],
            "Integración separa fallos de CNN y de política",
            errores,
        )
    else:
        registrar_aviso(False, f"Falta {ruta_integracion}", avisos)

    ruta_evaluacion = "notebooks/entrega3/03_evaluacion_rl.ipynb"
    if (RAIZ / ruta_evaluacion).is_file():
        resumen = resumir_notebook(cargar_notebook(ruta_evaluacion))
        registrar(
            resumen["celdas_ejecutadas"] == resumen["celdas_codigo"],
            "Todas las celdas de evaluación están ejecutadas",
            errores,
        )
        registrar(not resumen["errores"], "Evaluación no contiene errores guardados", errores)
        registrar("SEMILLA = 1000" in resumen["codigo"], "Evaluación inicia en la semilla 1000", errores)
        registrar("EPISODIOS_EVALUACION = 50" in resumen["codigo"], "Evaluación usa 50 episodios", errores)
        nombres_comunes = [
            "comparacion_agentes.csv",
            "configuraciones_hiperparametros.csv",
            "curva_recompensa.png",
            "comparacion_tasa_exito.png",
            "analisis_fallos.csv",
        ]
        registrar(
            all(nombre in resumen["codigo"] for nombre in nombres_comunes),
            "Evaluación exporta todos los nombres comunes",
            errores,
        )
    else:
        registrar_aviso(False, f"Falta {ruta_evaluacion}", avisos)

    ruta_final = "notebooks/entrega3/E3_StyleBot.ipynb"
    if (RAIZ / ruta_final).is_file():
        resumen = resumir_notebook(cargar_notebook(ruta_final))
        registrar_aviso(
            resumen["celdas_codigo"] > 0,
            "E3_StyleBot.ipynb todavía es una estructura sin código integrado",
            avisos,
        )
    else:
        registrar_aviso(False, f"Falta {ruta_final}", avisos)

    return errores, avisos


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
    errores_componentes, avisos = validar_componentes_nuevos()
    errores.extend(errores_componentes)

    print("\nResumen")
    if errores:
        print(f"La integración común tiene {len(errores)} error(es) que deben corregirse.")
    else:
        print("El entorno actual es compatible con el protocolo común de la Entrega 3.")
    print(f"Archivos finales pendientes: {len(faltantes)} de {len(ENTREGABLES_FINALES)}.")
    print(f"Avisos de integración: {len(avisos)}.")

    if errores or (argumentos.estricto and (faltantes or avisos)):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
