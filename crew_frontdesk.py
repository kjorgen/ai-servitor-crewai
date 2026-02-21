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
        model="gpt-4o-mini",
        api_key=os.getenv("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"),
        temperature=0.2
    )

    frontdesk = Agent(
        role="Frontdesk-agent",
        goal="Hjelpe restaurantgjester raskt og høflig på norsk.",
        backstory=(
            "Du er en robot-frontdesk for en norsk restaurant. "
            "Du svarer kort, konkret og serviceorientert. "
            "Hvis du mangler info, spør ett oppklarende spørsmål."
        ),
        llm=llm,
        verbose=False
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
