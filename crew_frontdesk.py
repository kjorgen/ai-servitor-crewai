import os
from crewai import Agent, Task, Crew
from crewai.llm import LLM

from pathlib import Path

KB_PATH = Path(__file__).with_name("knowledge_base.txt")

def load_kb() -> str:
    try:
        return KB_PATH.read_text(encoding="utf-8")
    except Exception:
        return ""

def run_frontdesk(message: str) -> str:
    # LLM (enkelt oppsett)
    llm = LLM(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        temperature=0.2,
    )

   kb = load_kb()

frontdesk_agent = Agent(
    role="Digital resepsjonist",
    goal="Svar korrekt på spørsmål om restauranten.",
    backstory=f"""
Du er en digital resepsjonist for restauranten Made in India.

DU MÅ følge disse reglene:

1. Svar kun basert på informasjonen i kunnskapsbasen.
2. Hvis informasjon ikke finnes i kunnskapsbasen:
   - Si at du ikke har informasjonen.
   - Ikke gjett.
3. Ikke finn på åpningstider, priser, allergener eller kontaktinfo.
4. Ved reservasjon skal du alltid be om:
   - dato
   - tidspunkt
   - antall personer
   - navn
   - telefonnummer

KUNNSKAPSBASEN:
{kb}
""",
    verbose=True
)
    )

    task = Task(
        description=(
            "Svar på meldingen fra kunden:\n"
            f"---\n{message}\n---\n\n"
            "Regler:\n"
            "- Svar på norsk\n"
            "- Maks 3 korte setninger\n"
            "- Hvis det handler om booking: be om dato, tidspunkt, antall personer\n"
            "- Hvis det handler om allergi: be om hvilken rett/allergi, og anbefal å dobbeltsjekke med restauranten\n"
            "- Ikke finn på åpningstider/adresse hvis du ikke vet"
        ),
        expected_output="Kort svar til kunden.",
        agent=frontdesk
    )

    crew = Crew(agents=[frontdesk], tasks=[task], verbose=False)
    result = crew.kickoff()
    return str(result).strip()
