from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from datetime import date
import requests
import json
import os
import io
import time
from pypdf import PdfReader
from supabase import create_client, Client

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONEXIÓN A SUPABASE ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# Lista cerrada de 16 arquetipos oficiales + categoría "Rol híbrido"
# ("Prompt Engineering" es competencia transversal, no arquetipo)
ARQUETIPOS_VALIDOS = [
    "Data Analyst",
    "Data Engineer",
    "Data Scientist",
    "ML Engineer",
    "MLOps Engineer",
    "AI Engineer",
    "Agentic AI Engineer",
    "Context Engineer",
    "AI Evaluation Engineer",
    "AI Safety / Red Team Engineer",
    "AI Architect",
    "AI Platform / Infrastructure Engineer",
    "AI Product Manager",
    "AI Consultant / Solutions Architect",
    "Forward Deployed Engineer",
    "AI Enablement Specialist",
    "Rol híbrido",
]

PESOS_ICR = {
    "definicion_funcional": 0.25,
    "coherencia_seniority": 0.15,
    "alineacion_titulo": 0.20,
    "especificidad_tecnica": 0.25,
    "entregables": 0.15,
}

def _clamp_0_100(valor, default: int = 0) -> int:
    try:
        num = int(round(float(valor)))
        return max(0, min(100, num))
    except (TypeError, ValueError):
        return default

def _calcular_nivel(score: int) -> str:
    if score >= 80:
        return "Rol claro y especializado"
    elif score >= 55:
        return "Rol comprensible con ambigüedad moderada"
    elif score >= 30:
        return "Rol híbrido o mal delimitado"
    else:
        return "Descripción confusa o incoherente"

def _normalizar_arquetipo(arquetipo_raw: Optional[str], es_hibrido: bool) -> str:
    if es_hibrido:
        return "Rol híbrido"
    if not arquetipo_raw:
        return "Rol híbrido"
    limpio = arquetipo_raw.strip()
    for valido in ARQUETIPOS_VALIDOS:
        if valido.lower() == limpio.lower():
            return valido
    return "Rol híbrido"

def _extraer_json(texto: str) -> dict:
    limpio = (texto or "").strip()
    if limpio.startswith("```"):
        lineas = limpio.splitlines()
        if lineas and lineas[0].startswith("```"):
            lineas = lineas[1:]
        if lineas and lineas[-1].strip() == "```":
            lineas = lineas[:-1]
        limpio = "\n".join(lineas).strip()

    try:
        return json.loads(limpio)
    except json.JSONDecodeError:
        inicio = limpio.find("{")
        fin = limpio.rfind("}")
        if inicio != -1 and fin != -1 and fin > inicio:
            return json.loads(limpio[inicio:fin + 1])
        raise

def call_gemini(prompt: str):
    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="API Key de Gemini no configurada")

    base = "https://" + "generativelanguage.googleapis.com"
    ruta = "/v1beta/models/gemini-flash-lite-latest:generateContent"
    url = base + ruta

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    max_intentos = 3
    for intento in range(max_intentos):
        response = requests.post(url, headers=headers, json=payload, timeout=25)

        if response.status_code == 200:
            result = response.json()
            texto = result["candidates"][0]["content"]["parts"][0]["text"]
            return _extraer_json(texto)

        if response.status_code in (429, 503) and intento < max_intentos - 1:
            time.sleep(1.5 * (intento + 1))
            continue

        raise HTTPException(
            status_code=response.status_code,
            detail=f"Error Gemini ({response.status_code}): {response.text}",
        )

def anonimizar_cv(cv_texto: str) -> str:
    prompt = f"""
    Eres un módulo de anonimización de datos personales (PII) para procesos de selección.
    Reescribe el siguiente texto de CV eliminando o reemplazando cualquier dato que identifique directamente a la persona:
    - Nombre y apellidos (usa "[CANDIDATO]")
    - Correos electrónicos, teléfonos, direcciones físicas, documentos de identidad o enlaces personales directos
    - Cualquier mención explícita al nombre del candidato en el cuerpo del texto

    Conserva con total fidelidad: experiencia laboral, roles, empresas (si no revelan identidad personal directa), tecnologías, proyectos, educación, logros y fechas.
    No agregues comentarios ni explicaciones. Devuelve SOLO un JSON válido con esta estructura:
    {{
      "cv_anonimizado": "..."
    }}

    CV ORIGINAL:
    {cv_texto}
    """
    data = call_gemini(prompt)
    anon = str(data.get("cv_anonimizado", "")).strip()
    return anon if anon else cv_texto

@app.post("/api/evaluate-job")
async def evaluate_job(
    title: str = Form(...),
    description: str = Form(...),
    empresa: Optional[str] = Form(None),
    modalidad: Optional[str] = Form(None),
    fuente: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    seniority_declarado: Optional[str] = Form(None),
    icr_manual: Optional[int] = Form(None),
    arquetipo_manual: Optional[str] = Form(None),
    notas: Optional[str] = Form(None),
):
    lista_arquetipos_str = ", ".join(f'"{a}"' for a in ARQUETIPOS_VALIDOS)

    prompt = f"""
    Eres un experto en reclutamiento técnico de IA y análisis organizacional.
    Analiza la siguiente oferta de trabajo y evalúa cada dimensión del Índice de Claridad del Rol de IA (ICR-IA) en una escala entera de 0 a 100:

    TÍTULO: {title}
    SENIORITY DECLARADO: {seniority_declarado or "No especificado"}
    DESCRIPCIÓN: {description}

    Dimensiones a puntuar (0 a 100):
    1. "definicion_funcional": Qué tan claro es el problema de negocio o producto que el rol debe resolver y su alcance funcional.
    2. "coherencia_seniority": Qué tan realista y coherente es la experiencia exigida frente a la complejidad técnica y autonomía pedida.
    3. "alineacion_titulo": Qué tan alineado está el título del puesto con las tareas y responsabilidades descritas.
    4. "especificidad_tecnica": Qué tan específicas son las tecnologías, modelos, APIs, datos o arquitecturas (vs. términos vagos o "buzzwords").
    5. "entregables": Qué tan claros, medibles y verificables son los resultados o entregables esperados del puesto.

    Clasificación de arquetipo:
    Elige ÚNICAMENTE uno de estos valores exactos: [{lista_arquetipos_str}].
    Usa "Rol híbrido" cuando la oferta mezcle responsabilidades dispares o múltiples perfiles sin un foco principal claro.
    Ten en cuenta que "Prompt Engineering" es una competencia transversal y NO un arquetipo de esta lista.
    Indica en "es_hibrido" (booleano) si el rol combina múltiples funciones o perfiles.

    En "feedback", resume en español claro: fortalezas de claridad, vacíos o ambigüedades detectadas y qué mejorar en la vacante.

    Devuelve SOLO un objeto JSON válido con esta estructura exacta:
    {{
      "definicion_funcional": 0,
      "coherencia_seniority": 0,
      "alineacion_titulo": 0,
      "especificidad_tecnica": 0,
      "entregables": 0,
      "es_hibrido": false,
      "arquetipo": "AI Engineer",
      "feedback": "..."
    }}
    """

    raw = call_gemini(prompt)

    dims = {
        "definicion_funcional": _clamp_0_100(raw.get("definicion_funcional")),
        "coherencia_seniority": _clamp_0_100(raw.get("coherencia_seniority")),
        "alineacion_titulo": _clamp_0_100(raw.get("alineacion_titulo")),
        "especificidad_tecnica": _clamp_0_100(raw.get("especificidad_tecnica")),
        "entregables": _clamp_0_100(raw.get("entregables")),
    }

    icr_score = int(round(sum(dims[k] * PESOS_ICR[k] for k in PESOS_ICR)))
    es_hibrido = bool(raw.get("es_hibrido", False))
    arquetipo = _normalizar_arquetipo(raw.get("arquetipo"), es_hibrido)
    if arquetipo == "Rol híbrido":
        es_hibrido = True

    nivel = _calcular_nivel(icr_score)
    feedback = str(raw.get("feedback", "")).strip()

    # GUARDAR EN SUPABASE
    if supabase:
        try:
            payload_db = {
                "titulo": title,
                "descripcion": description,
                "icr_score": icr_score,
                "arquetipo": arquetipo,
                "feedback": feedback,
                "empresa": empresa,
                "modalidad": modalidad,
                "fuente": fuente,
                "url": url,
                "fecha_consulta": date.today().isoformat(),
                "seniority_declarado": seniority_declarado,
                "dim_definicion_funcional": dims["definicion_funcional"],
                "dim_coherencia_seniority": dims["coherencia_seniority"],
                "dim_alineacion_titulo": dims["alineacion_titulo"],
                "dim_especificidad_tecnica": dims["especificidad_tecnica"],
                "dim_entregables": dims["entregables"],
                "nivel": nivel,
                "es_hibrido": es_hibrido,
                "icr_manual": icr_manual,
                "arquetipo_manual": arquetipo_manual,
                "notas": notas,
            }
            supabase.table("evaluacion_puestos").insert(payload_db).execute()
        except Exception as db_error:
            print(f"Error guardando puesto en Supabase: {db_error}")

    return {
        "icr_score": icr_score,
        "nivel": nivel,
        "arquetipo": arquetipo,
        "es_hibrido": es_hibrido,
        "feedback": feedback,
        "dimensiones": dims,
    }

@app.post("/api/analyze-cv")
async def analyze_cv(file: UploadFile = File(...), job_description: str = Form(...)):
    try:
        # 1. Leer el PDF
        contents = await file.read()
        pdf_reader = PdfReader(io.BytesIO(contents))
        cv_text = ""
        for page in pdf_reader.pages:
            cv_text += page.extract_text() + "\n"

        if not cv_text.strip():
            raise HTTPException(status_code=400, detail="No se pudo extraer texto del PDF")

        # 1.5 Sanitizar / anonimizar el CV antes de evaluarlo
        cv_limpio = anonimizar_cv(cv_text)

        # 2. Prompt para Gemini usando el CV ya anonimizado
        prompt = f"""
        Eres un sistema experto en reclutamiento técnico de IA.
        El CV ya fue anonimizado previamente; no incluyas datos personales ni nombres reales en tu respuesta.
        Descripción del puesto: {job_description}
        CV anonimizado: {cv_limpio}

        1. Calcula un Match Score del 0 al 100.
        2. Redacta un análisis técnico objetivo basado únicamente en las competencias y experiencia del CV.
        3. Genera 3 preguntas de entrevista técnica.

        Devuelve un JSON con: "match_score" (entero), "analisis" (string), "preguntas" (array de strings).
        SOLO EL JSON.
        """

        resultado = call_gemini(prompt)

        # 3. GUARDAR EN SUPABASE
        if supabase:
            try:
                supabase.table("analisis_cv").insert({
                    "candidate_name": "Candidato Anónimo",
                    "position": "AI Engineer",
                    "match_score": resultado.get("match_score", 0),
                    "analisis": resultado.get("analisis", ""),
                    "preguntas": json.dumps(resultado.get("preguntas", []))
                }).execute()
            except Exception as db_error:
                print(f"Error guardando en Supabase: {db_error}")

        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "API de AI Recruiter Copilot funcionando correctamente"}
