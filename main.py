import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# Henter API-nøkkel fra miljøvariabel
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

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


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Tar imot en melding fra brukeren og svarer som AI-servitør."""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.message},
        ],
        temperature=0.4,
    )

    reply = completion.choices[0].message.content.strip()
    return ChatResponse(reply=reply)
