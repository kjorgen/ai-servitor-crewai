from crew_frontdesk import run_frontdesk
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

from typing import Dict, Any, List
import re

SESSIONS: Dict[str, Dict[str, Any]] = {}

def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "history": [],   # list of {"role": "user"/"assistant", "text": "..."}
            "slots": {       # det agenten trenger for booking
                "date": None,
                "time": None,
                "people": None,
                "name": None,
                "phone": None,
            }
        }
    return SESSIONS[session_id]

def extract_slots(text: str, slots: Dict[str, Any]) -> None:
    t = text.strip()

    # antall personer
    m = re.search(r"\b(\d{1,2})\s*(person|pers|stk|gjester)\b", t, re.IGNORECASE)
    if m and not slots.get("people"):
        slots["people"] = m.group(1)

    # telefon (enkelt: 8 siffer, eller +47)
    m = re.search(r"(\+47)?\s*(\d{8})\b", t)
    if m and not slots.get("phone"):
        slots["phone"] = (m.group(1) or "") + m.group(2)

    # dato (super-enkel: "22 februar" / "22.02" etc. Du kan forbedre senere)
    m = re.search(r"\b(\d{1,2})[.\s]?(jan|feb|mar|apr|mai|jun|jul|aug|sep|okt|nov|des|januar|februar|mars|april|juni|juli|august|september|oktober|november|desember)\b", t, re.IGNORECASE)
    if m and not slots.get("date"):
        slots["date"] = m.group(0)

    m = re.search(r"\b(\d{1,2})\.(\d{1,2})\b", t)
    if m and not slots.get("date"):
        slots["date"] = m.group(0)

    # klokkeslett
    m = re.search(r"\bkl\.?\s*(\d{1,2})([:.](\d{2}))?\b", t, re.IGNORECASE)
    if m and not slots.get("time"):
        hh = m.group(1)
        mm = m.group(3) or "00"
        slots["time"] = f"{hh}:{mm}"

    m = re.search(r"\b(\d{1,2})[:.](\d{2})\b", t)
    if m and not slots.get("time"):
        slots["time"] = m.group(0).replace(".", ":")

    # navn (veldig enkel: "jeg heter X", "navn X")
    m = re.search(r"\b(jeg heter|navn)\s+([A-ZÆØÅ][a-zæøå]+(?:\s+[A-ZÆØÅ][a-zæøå]+)?)\b", t)
    if m and not slots.get("name"):
        slots["name"] = m.group(2)

def build_context(session: Dict[str, Any], max_turns: int = 8) -> str:
    # kort historikk
    turns = session["history"][-max_turns:]
    convo = "\n".join([f'{x["role"]}: {x["text"]}' for x in turns])

    slots = session["slots"]
    known = ", ".join([f"{k}={v}" for k, v in slots.items() if v])
    missing = [k for k, v in slots.items() if not v]

    return f"""
KJENT INFO (fra samtalen):
{known if known else "ingen"}

MANGLER (spør kun om dette hvis det trengs):
{", ".join(missing)}

KORT SAMTALEHISTORIKK:
{convo if convo else "ingen"}
""".strip()

# Henter API-nøkkel fra miljøvariabel
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#if not OPENAI_API_KEY:
    #raise RuntimeError("OPENAI_API_KEY is not set")

#client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="AI-servitør API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kjorgen.github.io",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: str
    message: str


SYSTEM_PROMPT = """
Du er en hjelpsom AI-servitør for en norsk restaurant.
Svar kort, høflig og konkret på norsk.
Du kan svare på:
- bordbestilling (uten å faktisk booke)
- åpningstider
- meny og anbefalinger (bruk eksempler hvis du ikke har ekte meny)
- allergier på et generelt nivå (anbefal alltid å spørre betjeningen ved tvil).

Hvis du ikke vet noe sikkert, si det ærlig.
Ikke finn opp detaljer om priser eller spesifikke råvarer.
"""


@app.get("/health")
def health():
    return {"status": "ok"}

from typing import List, Dict

def detect_intents(text: str) -> List[str]:
    t = text.lower().strip()
    intents = set()

    if any(k in t for k in ["book", "bestille", "bord", "reserver", "ledig"]):
        intents.add("BOOKING")

    if any(k in t for k in ["allerg", "gluten", "laktose", "nøtt", "peanøtt", "spor av", "cøliaki"]):
        intents.add("ALLERGY")

    if any(k in t for k in ["åpent", "åpning", "stenger", "adresse", "telefon", "parkering", "hvor ligger"]):
        intents.add("INFO")

    if any(k in t for k in ["meny", "rett", "retter", "pris", "anbefal", "drikke"]):
        intents.add("MENU")

    if not intents:
        intents.add("FALLBACK")

    ordered = []
    for x in ["ALLERGY", "BOOKING", "INFO", "MENU", "FALLBACK"]:
        if x in intents:
            ordered.append(x)

    return ordered


def allergy_agent(_: str) -> str:
    return (
        "Når det gjelder allergener kan jeg ikke garantere 100 %. "
        "Fortell gjerne hvilken allergi det gjelder, eller ta kontakt med restauranten direkte for sikkerhets skyld."
    )

def booking_agent(_: str) -> str:
    return "Jeg kan hjelpe med booking. Hvilken dato, klokkeslett og antall personer gjelder det?"

def info_agent(_: str) -> str:
    return "Jeg kan hjelpe med åpningstider, adresse og kontaktinfo. Hva lurer du på?"

def menu_agent(_: str) -> str:
    return "Jeg kan hjelpe med meny og anbefalinger. Hva slags mat er du ute etter?"

def fallback_agent(_: str) -> str:
    return "Kan du si litt mer konkret hva du trenger hjelp med?"


from crew_frontdesk import run_frontdesk  # antar du har denne

@app.post("/chat")
def chat(req: ChatRequest):
    session = get_session(req.session_id)

    # 1) lagre user-melding i historikk
    session["history"].append({"role": "user", "text": req.message})

    # 2) trekk ut “slots” fra user-meldingen
    extract_slots(req.message, session["slots"])

    # 3) bygg kontekst til agenten
    context = build_context(session)

    # 4) kall agenten med message + context
    reply = run_frontdesk(message=req.message, context=context)

    # 5) lagre svar
    session["history"].append({"role": "assistant", "text": reply})

    return {"reply": reply}

