from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import os
import io
from pypdf import PdfReader

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Función auxiliar para no repetir el código de la llamada a Gemini
def call_gemini(prompt: str, text_data: str = None):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API Key no configurada en Vercel")

    # Usamos el modelo flash-lite de tu proyecto anterior
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={api_key}"
    
    # Preparamos el contenido
    parts = [{"text": prompt}]
    if text_data:
        parts.append({"text": text_data})

    payload = {"contents": [{"parts": parts}]}
    headers = {'Content-Type': 'application/json'}
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        texto = result['candidates'][0]['content']['parts'][0]['text']
        texto = texto.strip()
        # Limpiamos los bloqueos de código Markdown
        if texto.startswith("```json"):
            texto = texto[7:-3].strip()
        elif texto.startswith("```"):
            texto = texto[3:-3].strip()
        return json.loads(texto)
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)


@app.post("/api/evaluate-job")
async def evaluate_job(title: str = Form(...), description: str = Form(...)):
    try:
        prompt = f"""
        Eres un experto en reclutamiento de IA. Analiza esta oferta de trabajo:
        TÍTULO: {title}
        DESCRIPCIÓN: {description}

        Tu tarea es calcular el Índice de Claridad del Rol de IA (ICR-IA) del 0 al 100.
        Si la descripción es muy corta o ambigua (ej. pide muchas cosas sin sentido), el score baja.
        Si es detallada y clara, el score sube.

        Devuelve un JSON estrictamente estructurado con estas claves:
        - "icr_score": (número entero)
        - "arquetipo": (string, ej. "AI Engineer", "Data Scientist")
        - "feedback": (string, una recomendación corta para el reclutador)
        
        Devuelve SOLO el JSON, sin texto extra.
        """
        
        resultado = call_gemini(prompt)
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-cv")
async def analyze_cv(file: UploadFile = File(...), job_description: str = Form(...)):
    try:
        # 1. Leer el PDF y extraer el texto
        contents = await file.read()
        pdf_reader = PdfReader(io.BytesIO(contents))
        cv_text = ""
        for page in pdf_reader.pages:
            cv_text += page.extract_text() + "\n"
            
        if not cv_text.strip():
            raise HTTPException(status_code=400, detail="No se pudo extraer texto del PDF")

        # 2. Prompt Maestro (Anonimización + Match Score + Preguntas)
        prompt = f"""
        Eres un sistema experto en reclutamiento técnico de IA.
        Tienes la siguiente descripción del puesto:
        {job_description}

        Y tienes este Currículum Vitae (CV):
        {cv_text}

        Sigue estas instrucciones ESTRICTAMENTE:
        1. Anonimiza el CV mentalmente (ignora nombre, edad, género, etc.).
        2. Calcula un Match Score del 0 al 100 comparando las habilidades reales del CV con lo que pide el puesto.
        3. Genera 3 preguntas de entrevista técnica basadas en el CV para un reclutador no técnico.

        Devuelve un JSON estrictamente estructurado con estas claves:
        - "match_score": (número entero)
        - "analisis": (string con fortalezas y debilidades)
        - "preguntas": (array de 3 strings con las preguntas)
        
        Devuelve SOLO el JSON, sin texto extra.
        """
        
        resultado = call_gemini(prompt)
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"status": "API de AI Recruiter Copilot funcionando correctamente"}
