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

def run_frontdesk(
    message: str,
    context: str = "",
    history: list[dict] | None = None
) -> str:
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
    [f"{m.get('role','user')}: {m.get('text') or m.get('content') or ''}" for m in history_trimmed]
).strip()

rules = """
REGLER (MÅ FØLGES):
1) Svar kun basert på KUNNSKAPSBASEN og KONTEKSTEN fra denne chatten.
2) Hvis info ikke finnes i kunnskapsbasen eller konteksten: si at du ikke har informasjonen. Ikke gjett.
3) Ikke finn på åpningstider, priser, allergener, kontaktinfo eller bookingdetaljer.
4) Ved reservasjon: bruk først info som allerede finnes i KONTEKSTEN.
5) Ikke be om informasjon som allerede er oppgitt.
6) Still MAKS 1 oppfølgingsspørsmål om gangen.
7) For booking: spør i denne rekkefølgen: dato -> tidspunkt -> antall personer -> navn -> telefon.
8) Hvis KONTEKSTEN viser hva som mangler, skal du kun spørre om neste manglende felt.
9) Hvis alle bookingfelter er fylt ut, skal du ikke spørre om mer informasjon.
10) Ikke bland inn allergi, meny eller andre temaer i en bookingflyt med mindre brukeren spør om det.
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
        "VIKTIG:\n"
        "- Bruk KONTEKSTEN under som fasit for hva som allerede er kjent.\n"
        "- Hvis KONTEKSTEN sier at noe allerede er oppgitt, skal du ikke spørre om det igjen.\n"
        "- Hvis KONTEKSTEN viser manglende bookingfelter, skal du kun spørre om NESTE manglende felt.\n"
        "- Still maks ett spørsmål i svaret.\n"
        "- Ikke bland inn andre temaer hvis brukeren er i bookingflyt.\n\n"
        f"KONTEKST (kjent info + mangler + kort historikk):\n{context}\n\n"
        f"EKSTRA HISTORIKK (rå, siste 10):\n{history_text or 'ingen'}\n\n"
        f"NY MELDING:\n{message}\n\n"
        "KRAV TIL SVARET:\n"
        "- Svar på norsk\n"
        "- Maks 2 korte setninger\n"
        "- Ved booking: spør kun om ett manglende felt\n"
        "- Ved allergi: svar forsiktig og anbefal dobbeltsjekk med restauranten ved tvil\n"
        "- Ikke finn på informasjon som ikke finnes i kunnskapsbasen eller konteksten\n"
    ),
    expected_output="Et kort, korrekt svar på norsk med maks ett oppfølgingsspørsmål.",
    agent=frontdesk,
)

    crew = Crew(agents=[frontdesk], tasks=[task], verbose=False)
    result = crew.kickoff()
    return str(result).strip()
