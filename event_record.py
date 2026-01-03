from dataclasses import dataclass
from typing import Optional


@dataclass
class EventRecord:
    event_type: str
    who: Optional[str]
    what: Optional[str]
    trigger: Optional[str]
    where: Optional[str] = None
    when: Optional[str] = None
    confidence: float = 0.0
    sentence: str = ""

    def __str__(self) -> str:
        lines = [f"Typ zdarzenia: {self.event_type}"]

        if self.who:
            lines.append(f"KTO: {self.who}")
        if self.what:
            lines.append(f"CO: {self.what}")
        if self.trigger:
            lines.append(f"Trigger: {self.trigger}")
        if self.where:
            lines.append(f"GDZIE: {self.where}")
        if self.when:
            lines.append(f"KIEDY: {self.when}")

        lines.append(f"Pewność: {self.confidence:.2f}")
        return "\n".join(lines)
