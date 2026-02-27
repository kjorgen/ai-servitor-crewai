import os
from pathlib import Path

from crewai import Agent, Task, Crew
from crewai.llm import LLM

KB_PATH = Path(__file__).with_name("knowledge_base.txt")

def load_kb() -> str:
    try:
        return KB_PATH.read_text(encoding="utf-8")
    except Exception:
        return ""

def run_frontdesk(message: str, history: list[dict] | None = None) -> str:
    """
    history: liste med meldinger som [{"role":"user|assistant","content":"..."}]
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Beklager, serveren mangler API-nøkkel (OPENAI_API_KEY)."

    llm = LLM(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=0.2,
    )

    kb = load_kb()

    # Bygg en enkel kontekststreng fra historikk (siste 10 meldinger)
    history = history or []
    history_trimmed = history[-10:]
    history_text = "\n".join(
        [f"{m.get('role','user')}: {m.get('content','')}" for m in history_trimmed]
    ).strip()

    rules = """
REGLER (MÅ FØLGES):
1) Svar kun basert på KUNNSKAPSBASEN og det brukeren har oppgitt i denne chatten.
2) Hvis info ikke finnes i kunnskapsbasen: si at du ikke har informasjonen. Ikke gjett.
3) Ikke finn på åpningstider, priser, allergener eller kontaktinfo.
4) Ved reservasjon: be om det som mangler av: dato, tidspunkt, antall personer, navn, telefonnummer.
5) Hvis kunden allerede har gitt noe informasjon tidligere i chatten, IKKE be om det på nytt. Bruk det som er oppgitt.
"""

    frontdesk = Agent(
        role="Digital resepsjonist",
        goal="Svar korrekt på spørsmål om restauranten, og samle inn reservasjon-detaljer på en ryddig måte.",
        backstory=(
            "Du er en digital resepsjonist for restauranten Made in India.\n\n"
            f"{rules}\n\n"
            "KUNNSKAPSBASEN:\n"
            f"{kb}\n"
        ),
        llm=llm,
        verbose=False,
    )

    task = Task(
        description=(
            "Du svarer på en melding fra kunden.\n\n"
            f"KONTEKST (tidligere meldinger i chatten):\n{history_text}\n\n"
            f"NY MELDING:\n{message}\n\n"
            "Krav:\n"
            "- Svar på norsk\n"
            "- Maks 3 korte setninger\n"
            "- Ved booking: be kun om det som mangler (dato/tid/antall/navn/tlf)\n"
            "- Ved allergi: be om hvilken rett/allergi og anbefal dobbeltsjekk med restauranten\n"
            "- Ikke finn på informasjon som ikke finnes i kunnskapsbasen\n"
        ),
        expected_output="Kort, korrekt svar til kunden.",
        agent=frontdesk,
    )

    crew = Crew(agents=[frontdesk], tasks=[task], verbose=False)
    result = crew.kickoff()
    return str(result).strip()
