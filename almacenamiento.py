"""
Módulo de almacenamiento y persistencia para Paddock Global (Fórmula 1).
Cumple con las especificaciones de persistencia atómica, validación de esquemas y consultas en memoria.
"""

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class ErrorAlmacenamiento(Exception):
    """Excepción base para errores de persistencia o lectura."""
    pass


def cargar_pilotos(ruta: Path | str) -> List[Dict[str, Any]]:
    """
    Lee y valida el catálogo de pilotos desde un archivo JSON.
    Comprueba versión, estructura y que no existan identificadores repetidos.
    """
    ruta_path = Path(ruta)
    if not ruta_path.exists():
        raise FileNotFoundError(f"El archivo de catálogo no existe en: {ruta_path}")

    try:
        with open(ruta_path, "r", encoding="utf-8") as f:
            datos = json.load(f)
    except json.JSONDecodeError as e:
        raise ErrorAlmacenamiento(f"El archivo de catálogo contiene JSON inválido: {e}")

    if not isinstance(datos, dict) or "pilotos" not in datos or not isinstance(datos["pilotos"], list):
        raise ErrorAlmacenamiento("Estructura inválida en el catálogo de pilotos (se esperaba objeto con clave 'pilotos').")

    ids_vistos = set()
    pilotos_validados = []
    for p in datos["pilotos"]:
        if not isinstance(p, dict) or "id" not in p or "nombre" not in p:
            raise ErrorAlmacenamiento(f"Piloto mal estructurado en el catálogo: {p}")
        p_id = str(p["id"]).strip()
        if p_id in ids_vistos:
            raise ErrorAlmacenamiento(f"Identificador duplicado detectado en el catálogo de pilotos: {p_id}")
        ids_vistos.add(p_id)
        pilotos_validados.append(p)

    return pilotos_validados


def cargar_opiniones(ruta: Path | str) -> Dict[str, Any]:
    """
    Lee el diccionario de opiniones persistentes.
    Si el archivo no existe, lo inicializa con estructura vacía.
    Si el archivo existe pero contiene JSON corrupto o esquema inesperado,
    lanza una excepción y conserva el archivo intacto (no lo sobrescribe).
    """
    ruta_path = Path(ruta)
    if not ruta_path.exists():
        estructura_inicial = {
            "version": 1,
            "opiniones": []
        }
        guardar_diccionario_completo(ruta_path, estructura_inicial)
        return estructura_inicial

    try:
        with open(ruta_path, "r", encoding="utf-8") as f:
            contenido = f.read().strip()
            if not contenido:
                # Archivo vacío por interrupción previa: inicializar
                estructura_inicial = {"version": 1, "opiniones": []}
                guardar_diccionario_completo(ruta_path, estructura_inicial)
                return estructura_inicial
            datos = json.loads(contenido)
    except json.JSONDecodeError as e:
        raise ErrorAlmacenamiento(f"El archivo de opiniones contiene JSON inválido o está dañado: {e}. Se conserva intacto para evitar pérdida de datos.")

    if not isinstance(datos, dict) or "opiniones" not in datos or not isinstance(datos["opiniones"], list):
        raise ErrorAlmacenamiento("El archivo de opiniones tiene un esquema inesperado. No se sobrescribirá automáticamente.")

    return datos


def crear_registro(piloto: Dict[str, Any], texto: str, analisis: Any, modelo: str, version_prompt: str = "1.0") -> Dict[str, Any]:
    """
    Construye el diccionario de registro con UUID, fecha UTC y metadatos del análisis.
    """
    sentimiento = getattr(analisis, "sentimiento", None)
    razon = getattr(analisis, "razon", "")
    if isinstance(analisis, dict):
        sentimiento = analisis.get("sentimiento")
        razon = analisis.get("razon", "")

    return {
        "id": str(uuid.uuid4()),
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "piloto_id": piloto.get("id"),
        "piloto_nombre": piloto.get("nombre"),
        "opinion_original": texto.strip(),
        "sentimiento": sentimiento,
        "razon": razon,
        "modelo": modelo,
        "version_prompt": version_prompt
    }


def guardar_diccionario_completo(ruta_path: Path, datos: Dict[str, Any]) -> None:
    """
    Escribe el diccionario completo de forma atómica usando un archivo temporal y reemplazo final.
    """
    ruta_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=ruta_path.parent, delete=False, encoding="utf-8") as tf:
            temp_file = tf.name
            json.dump(datos, tf, ensure_ascii=False, indent=2)
            tf.flush()
            os.fsync(tf.fileno())
        os.replace(temp_file, ruta_path)
    except Exception as e:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass
        raise ErrorAlmacenamiento(f"Fallo de escritura atómica en el archivo de opiniones: {e}")


def guardar_opinion(ruta: Path | str, registro: Dict[str, Any]) -> None:
    """
    Carga el estado existente, previene duplicados por UUID, agrega el nuevo registro
    y escribe de manera atómica sin perder los anteriores.
    """
    ruta_path = Path(ruta)
    datos = cargar_opiniones(ruta_path)
    
    # Prevenir duplicación en caso de reintentos con el mismo UUID
    reg_id = registro.get("id")
    ids_existentes = {op.get("id") for op in datos.get("opiniones", [])}
    if reg_id and reg_id in ids_existentes:
        return  # Ya se encuentra persistido

    datos["opiniones"].append(registro)
    guardar_diccionario_completo(ruta_path, datos)


def consultar_opiniones(
    opiniones: List[Dict[str, Any]],
    piloto_id: Optional[str] = None,
    sentimiento: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Filtra la lista de opiniones por piloto y sentimiento sin interactuar con el LLM.
    """
    resultados = []
    piloto_filtro = piloto_id.strip() if piloto_id else "todos"
    sentimiento_filtro = sentimiento.strip().lower() if sentimiento else "todos"

    for op in opiniones:
        coincide_piloto = (piloto_filtro == "todos" or op.get("piloto_id") == piloto_filtro)
        coincide_sentimiento = (sentimiento_filtro == "todos" or str(op.get("sentimiento", "")).lower() == sentimiento_filtro)

        if coincide_piloto and coincide_sentimiento:
            resultados.append(op)

    return resultados
