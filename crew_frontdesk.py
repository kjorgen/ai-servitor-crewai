import os
from pathlib import Path

from crewai import Agent, Task, Crew
from crewai.llm import LLM

KB_PATH = Path(__file__).with_name("knowledge_base.txt")


def load_kb() -> str:
    try:
        return KB_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def run_frontdesk(message: str) -> str:
    kb = load_kb()

    llm = LLM(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.2,
    )

    frontdesk = Agent(
        role="Digital resepsjonist",
        goal="Svar korrekt på spørsmål om restauranten.",
        backstory=f"""
Du er en digital resepsjonist for restauranten Made in India.

REGLER:
1) Svar kun basert på KUNNSKAPSBASEN.
2) Hvis info ikke finnes: si at du ikke har informasjonen.
3) Ikke gjett åpningstider, priser, allergener eller kontaktinfo.
4) Ved reservasjon: be alltid om dato, tidspunkt, antall personer, navn, telefonnummer.
5) Hvis kunden allerede har gitt noe informasjon i tidligere meldinger,
   skal du ikke be om det på nytt.
   
KUNNSKAPSBASEN:
{kb}
""".strip(),
        llm=llm,
        verbose=False,
    )

    task = Task(
        description=(
            "Svar på meldingen fra kunden:\n"
            f"{message}\n\n"
            "Krav:\n"
            "- Svar på norsk\n"
            "- Maks 3 korte setninger\n"
            "- Ved booking: be om dato, tidspunkt, antall personer, navn, telefonnummer\n"
            "- Ikke finn på informasjon som ikke finnes i kunnskapsbasen"
        ),
        expected_output="Kort svar til kunden.",
        agent=frontdesk,
    )

    crew = Crew(agents=[frontdesk], tasks=[task], verbose=False)
    result = crew.kickoff()

    return str(result).strip()
