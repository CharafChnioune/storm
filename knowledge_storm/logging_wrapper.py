from contextlib import contextmanager
import time
import pytz
from datetime import datetime

# Definieer Californië tijdzone
CALIFORNIA_TZ = pytz.timezone("America/Los_Angeles")


class EventLog:
    """
    Klasse voor het loggen van individuele gebeurtenissen met start- en eindtijden.
    Ondersteunt geneste gebeurtenissen via een boomstructuur.
    """

    def __init__(self, event_name):
        self.event_name = event_name
        self.start_time = None
        self.end_time = None
        self.child_events = {}

    def record_start_time(self):
        # Sla op in UTC voor consistente tijdzone-conversie
        self.start_time = datetime.now(pytz.utc)

    def record_end_time(self):
        # Sla op in UTC voor consistente tijdzone-conversie
        self.end_time = datetime.now(pytz.utc)

    def get_total_time(self):
        """Bereken de totale duur van de gebeurtenis in seconden."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0

    def get_start_time(self):
        """Geef de starttijd terug in Californië tijdzone, geformatteerd tot op milliseconden."""
        if self.start_time:
            # Formatteer tot milliseconden
            return self.start_time.astimezone(CALIFORNIA_TZ).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3]
        return None

    def get_end_time(self):
        """Geef de eindtijd terug in Californië tijdzone, geformatteerd tot op milliseconden."""
        if self.end_time:
            # Formatteer tot milliseconden
            return self.end_time.astimezone(CALIFORNIA_TZ).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3]
        return None

    def add_child_event(self, child_event):
        """Voeg een geneste gebeurtenis toe aan deze gebeurtenis."""
        self.child_events[child_event.event_name] = child_event

    def get_child_events(self):
        """Haal alle geneste gebeurtenissen op."""
        return self.child_events


class LoggingWrapper:
    """
    Wrapper klasse voor het loggen van pipelinefases en gebeurtenissen.
    Houdt tijdsgebruik, LM-gebruik en querytellingen bij.
    """

    def __init__(self, lm_config):
        self.logging_dict = {}
        self.lm_config = lm_config
        self.current_pipeline_stage = None
        self.event_stack = []
        self.pipeline_stage_active = False

    def _pipeline_stage_start(self, pipeline_stage: str):
        """Start een nieuwe pipelinefase en initialiseer de logboekstructuur."""
        if self.pipeline_stage_active:
            raise RuntimeError(
                "Er is al een pipelinefase actief. Beëindig de huidige fase voordat je een nieuwe start."
            )

        self.current_pipeline_stage = pipeline_stage
        self.logging_dict[pipeline_stage] = {
            "time_usage": {},
            "lm_usage": {},
            "lm_history": [],
            "query_count": 0,
        }
        self.pipeline_stage_active = True

    def _event_start(self, event_name: str):
        """Start een nieuwe gebeurtenis binnen de huidige pipelinefase of als geneste gebeurtenis."""
        if not self.pipeline_stage_active:
            raise RuntimeError("Er is momenteel geen pipelinefase actief.")

        if not self.event_stack and self.current_pipeline_stage:
            # Gebeurtenis op hoogste niveau (direct onder de pipelinefase)
            if (
                event_name
                not in self.logging_dict[self.current_pipeline_stage]["time_usage"]
            ):
                event = EventLog(event_name=event_name)
                event.record_start_time()
                self.logging_dict[self.current_pipeline_stage]["time_usage"][
                    event_name
                ] = event
                self.event_stack.append(event)
            else:
                self.logging_dict[self.current_pipeline_stage]["time_usage"][
                    event_name
                ].record_start_time()
        elif self.event_stack:
            # Geneste gebeurtenis (onder een andere gebeurtenis)
            parent_event = self.event_stack[-1]
            if event_name not in parent_event.get_child_events():
                event = EventLog(event_name=event_name)
                event.record_start_time()
                parent_event.add_child_event(event)
                self.logging_dict[self.current_pipeline_stage]["time_usage"][
                    event_name
                ] = event
                self.event_stack.append(event)
            else:
                parent_event.get_child_events()[event_name].record_start_time()
        else:
            raise RuntimeError(
                "Kan geen gebeurtenis starten zonder actieve pipelinefase of bovenliggende gebeurtenis."
            )

    def _event_end(self, event_name: str):
        """Beëindig een gebeurtenis en registreer de eindtijd."""
        if not self.pipeline_stage_active:
            raise RuntimeError("Er is momenteel geen pipelinefase actief.")

        if not self.event_stack:
            raise RuntimeError("Er is momenteel geen bovenliggende gebeurtenis actief.")

        if self.event_stack:
            current_event_log = self.event_stack[-1]
            if event_name in current_event_log.get_child_events():
                current_event_log.get_child_events()[event_name].record_end_time()
            elif (
                event_name
                in self.logging_dict[self.current_pipeline_stage]["time_usage"]
            ):
                self.logging_dict[self.current_pipeline_stage]["time_usage"][
                    event_name
                ].record_end_time()
            else:
                raise AssertionError(
                    f"Kan eindtijd voor gebeurtenis {event_name} niet registreren. Starttijd is niet geregistreerd."
                )
            if current_event_log.event_name == event_name:
                self.event_stack.pop()
        else:
            raise RuntimeError(
                "Kan een gebeurtenis niet beëindigen zonder actieve bovenliggende gebeurtenis."
            )

    def _pipeline_stage_end(self):
        """Beëindig de huidige pipelinefase en verzamel LM-gebruiksstatistieken."""
        if not self.pipeline_stage_active:
            raise RuntimeError("Er is momenteel geen actieve pipelinefase om te beëindigen.")

        self.logging_dict[self.current_pipeline_stage][
            "lm_usage"
        ] = self.lm_config.collect_and_reset_lm_usage()
        self.logging_dict[self.current_pipeline_stage][
            "lm_history"
        ] = self.lm_config.collect_and_reset_lm_history()
        self.pipeline_stage_active = False

    def add_query_count(self, count):
        """Voeg het aantal queries toe aan de huidige pipelinefase."""
        if not self.pipeline_stage_active:
            raise RuntimeError(
                "Er is momenteel geen actieve pipelinefase om querytellingen aan toe te voegen."
            )

        self.logging_dict[self.current_pipeline_stage]["query_count"] += count

    @contextmanager
    def log_event(self, event_name):
        """Context manager voor het loggen van een gebeurtenis."""
        if not self.pipeline_stage_active:
            raise RuntimeError("Er is momenteel geen pipelinefase actief.")

        self._event_start(event_name)
        yield
        self._event_end(event_name)

    @contextmanager
    def log_pipeline_stage(self, pipeline_stage):
        """Context manager voor het loggen van een pipelinefase."""
        if self.pipeline_stage_active:
            print(
                "Er is al een pipelinefase actief, de huidige fase wordt veilig beëindigd."
            )
            self._pipeline_stage_end()

        start_time = time.time()
        try:
            self._pipeline_stage_start(pipeline_stage)
            yield
        except Exception as e:
            print(f"Fout opgetreden tijdens pipelinefase '{pipeline_stage}': {e}")
        finally:
            self.logging_dict[self.current_pipeline_stage]["total_wall_time"] = (
                time.time() - start_time
            )
            self._pipeline_stage_end()

    def dump_logging_and_reset(self, reset_logging=True):
        """
        Dump alle logboekgegevens en reset optioneel de logboekstructuur.
        Retourneert een gestructureerde weergave van alle gelogde informatie.
        """
        log_dump = {}
        for pipeline_stage, pipeline_log in self.logging_dict.items():
            time_stamp_log = {
                event_name: {
                    "total_time_seconds": event.get_total_time(),
                    "start_time": event.get_start_time(),
                    "end_time": event.get_end_time(),
                }
                for event_name, event in pipeline_log["time_usage"].items()
            }
            log_dump[pipeline_stage] = {
                "time_usage": time_stamp_log,
                "lm_usage": pipeline_log["lm_usage"],
                "lm_history": pipeline_log["lm_history"],
                "query_count": pipeline_log["query_count"],
                "total_wall_time": pipeline_log["total_wall_time"],
            }
        if reset_logging:
            self.logging_dict.clear()
        return log_dump