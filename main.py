import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# Henter API-nøkkel fra miljøvariabel
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#if not OPENAI_API_KEY:
    #raise RuntimeError("OPENAI_API_KEY is not set")

#client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="AI-servitør API")

# Enkel CORS slik at GitHub-siden din får lov til å kalle API-et
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


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


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    message = req.message

    intents = detect_intents(message)
    parts = {}

    for intent in intents:
        if intent == "ALLERGY":
            parts[intent] = allergy_agent(message)
        elif intent == "BOOKING":
            parts[intent] = booking_agent(message)
        elif intent == "INFO":
            parts[intent] = info_agent(message)
        elif intent == "MENU":
            parts[intent] = menu_agent(message)
        else:
            parts[intent] = fallback_agent(message)

    reply = "\n\n".join(parts.values())
    return ChatResponse(reply=reply)

