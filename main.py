from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import os
import io
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

def call_gemini(prompt: str):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API Key de Gemini no configurada")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        texto = result['candidates'][0]['content']['parts'][0]['text']
        texto = texto.strip()
        # Limpiar formato Markdown de la respuesta
        if texto.startswith("```json"):
            texto = texto[7:-3].strip()
        elif texto.startswith("```"):
            texto = texto[3:-3].strip()
        return json.loads(texto)
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

@app.post("/api/evaluate-job")
async def evaluate_job(title: str = Form(...), description: str = Form(...)):
    prompt = f"""
    Eres un experto en reclutamiento de IA. Analiza esta oferta de trabajo:
    TÍTULO: {title}
    DESCRIPCIÓN: {description}
    Calcula el Índice de Claridad del Rol de IA (ICR-IA) del 0 al 100.
    Devuelve un JSON con: "icr_score" (entero), "arquetipo" (string), "feedback" (string).
    SOLO EL JSON.
    """
    return call_gemini(prompt)

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

        # 2. Prompt para Gemini
        prompt = f"""
        Eres un sistema experto en reclutamiento técnico de IA.
        Descripción del puesto: {job_description}
        CV: {cv_text}
        
        1. Anonimiza el CV mentalmente.
        2. Calcula un Match Score del 0 al 100.
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
                    "position": "AI Engineer", # Lo dejamos fijo por ahora
                    "match_score": resultado.get("match_score", 0),
                    "analisis": resultado.get("analisis", ""),
                    "preguntas": json.dumps(resultado.get("preguntas", [])) # Convertimos la lista a texto
                }).execute()
            except Exception as db_error:
                print(f"Error guardando en Supabase: {db_error}")
                # No detenemos la app si falla la base de datos, solo imprimimos el error

        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "API de AI Recruiter Copilot funcionando correctamente"}
