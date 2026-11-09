"""
Suite de pruebas unitarias para el sistema Paddock Global (Fórmula 1).
Verifica catálogo, contratos Pydantic, persistencia atómica, filtros y manejo de errores.
"""

import json
import pytest
from pathlib import Path
from pydantic import ValidationError

from almacenamiento import (
    cargar_pilotos,
    cargar_opiniones,
    crear_registro,
    guardar_opinion,
    consultar_opiniones,
    ErrorAlmacenamiento
)
from analisis import AnalisisOpinion, analizar_opinion

BASE_DIR = Path(__file__).resolve().parent
RUTA_PILOTOS = BASE_DIR / "datos" / "pilotos.json"
RUTA_OPINIONES = BASE_DIR / "datos" / "opiniones.json"


# --- 1. PRUEBAS DEL CATÁLOGO ---
def test_catalogo_pilotos_integridad():
    """El catálogo debe tener al menos 20 pilotos con IDs únicos y campos requeridos."""
    pilotos = cargar_pilotos(RUTA_PILOTOS)
    assert len(pilotos) >= 20, "El catálogo debe contener al menos 20 pilotos."
    
    ids = [p["id"] for p in pilotos]
    assert len(ids) == len(set(ids)), "Los identificadores de pilotos deben ser únicos."
    
    for p in pilotos:
        assert "id" in p and p["id"].startswith("p"), f"ID inválido en piloto: {p}"
        assert "nombre" in p and len(p["nombre"]) > 0
        assert "pais" in p
        assert "escuderia" in p


def test_catalogo_archivo_inexistente():
    """Debe lanzar FileNotFoundError si la ruta no existe."""
    with pytest.raises(FileNotFoundError):
        cargar_pilotos(BASE_DIR / "datos" / "no_existe.json")


# --- 2. PRUEBAS DE PERSISTENCIA Y ALMACENAMIENTO ---
def test_cargar_opiniones_existente():
    """Debe cargar el archivo de opiniones con estructura válida y metadatos correctos."""
    datos = cargar_opiniones(RUTA_OPINIONES)
    assert datos.get("version") == 1
    assert "opiniones" in datos
    assert len(datos["opiniones"]) >= 12, "Debe contener al menos 12 opiniones iniciales de prueba."
    
    primer_registro = datos["opiniones"][0]
    campos_obligatorios = {
        "id", "fecha_utc", "piloto_id", "piloto_nombre",
        "opinion_original", "sentimiento", "razon", "modelo", "version_prompt"
    }
    assert campos_obligatorios.issubset(primer_registro.keys()), "Faltan campos obligatorios en el registro."


def test_guardar_opinion_y_deduplicacion(tmp_path):
    """Verifica la persistencia atómica y la prevención de duplicados por UUID."""
    ruta_temp = tmp_path / "opiniones_test.json"
    
    piloto = {"id": "p001", "nombre": "Fernando Alonso"}
    analisis = AnalisisOpinion(
        es_pertinente=True,
        sentimiento="positivo",
        razon="Conducción sobresaliente bajo presión."
    )
    
    # 1. Crear registro
    registro = crear_registro(piloto, "Excelente ritmo de carrera en Suzuka", analisis, "test-model")
    assert registro["piloto_id"] == "p001"
    assert registro["sentimiento"] == "positivo"
    assert "T" in registro["fecha_utc"]  # Formato ISO UTC
    
    # 2. Guardar primera vez
    guardar_opinion(ruta_temp, registro)
    cargados = cargar_opiniones(ruta_temp)
    assert len(cargados["opiniones"]) == 1
    
    # 3. Intentar guardar el mismo registro (mismo UUID) -> No debe duplicarse
    guardar_opinion(ruta_temp, registro)
    cargados_despues = cargar_opiniones(ruta_temp)
    assert len(cargados_despues["opiniones"]) == 1, "No debe duplicar registros con el mismo UUID."


def test_archivo_corrupto_no_se_sobrescribe(tmp_path):
    """Si el archivo de opiniones tiene JSON inválido, no debe borrarse silenciosamente."""
    ruta_corrupta = tmp_path / "corrupto.json"
    ruta_corrupta.write_text("{ esto no es un json valido", encoding="utf-8")
    
    with pytest.raises(ErrorAlmacenamiento):
        cargar_opiniones(ruta_corrupta)
    
    # Verificar que el contenido no fue borrado
    assert ruta_corrupta.read_text(encoding="utf-8") == "{ esto no es un json valido"


# --- 3. PRUEBAS DE CONSULTA Y FILTRADO ---
def test_consultar_opiniones_filtros():
    """Verifica que los filtros por piloto y sentimiento operen en memoria correctamente."""
    datos = cargar_opiniones(RUTA_OPINIONES)
    todas = datos["opiniones"]
    
    # Filtro por piloto
    alonso_ops = consultar_opiniones(todas, piloto_id="p001")
    assert len(alonso_ops) > 0
    assert all(op["piloto_id"] == "p001" for op in alonso_ops)
    
    # Filtro por sentimiento
    pos_ops = consultar_opiniones(todas, sentimiento="positivo")
    assert len(pos_ops) > 0
    assert all(op["sentimiento"] == "positivo" for op in pos_ops)
    
    # Filtro cruzado (piloto + sentimiento)
    alonso_pos = consultar_opiniones(todas, piloto_id="p001", sentimiento="positivo")
    assert all(op["piloto_id"] == "p001" and op["sentimiento"] == "positivo" for op in alonso_pos)
    
    # Sin coincidencias
    vacio = consultar_opiniones(todas, piloto_id="p999")
    assert len(vacio) == 0


# --- 4. PRUEBAS DEL CONTRATO PYDANTIC ---
def test_analisis_opinion_pydantic_valido():
    """Valida la creación correcta de opiniones pertinentes."""
    valido = AnalisisOpinion(
        es_pertinente=True,
        sentimiento="mixto",
        razon="Tiene gran velocidad punta pero comete fallos al defender."
    )
    assert valido.es_pertinente is True
    assert valido.sentimiento == "mixto"


def test_analisis_opinion_no_pertinente_fuerza_none():
    """Una opinión no pertinente debe forzar sentimiento=None."""
    no_pertinente = AnalisisOpinion(
        es_pertinente=False,
        sentimiento="positivo",  # Aunque se envíe, el validator lo fuerza a None
        razon="El texto habla sobre tenis y no sobre el piloto."
    )
    assert no_pertinente.es_pertinente is False
    assert no_pertinente.sentimiento is None


def test_analisis_opinion_pertinente_sin_sentimiento_falla():
    """Una opinión marcada como pertinente sin sentimiento debe fallar la validación."""
    with pytest.raises(ValidationError):
        AnalisisOpinion(
            es_pertinente=True,
            sentimiento=None,
            razon="Elogio al piloto."
        )


def test_validacion_longitud_opinion():
    """Opiniones menores a 5 caracteres o mayores a 1000 deben rechazarse antes de llamar al modelo."""
    from unittest.mock import MagicMock
    mock_llm = MagicMock()
    
    with pytest.raises(ValueError, match="al menos 5 caracteres"):
        analizar_opinion(mock_llm, "Fernando Alonso", "abc")
        
    with pytest.raises(ValueError, match="no puede superar los 1000 caracteres"):
        analizar_opinion(mock_llm, "Fernando Alonso", "A" * 1001)
