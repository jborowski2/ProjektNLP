#!/usr/bin/env python3
"""
Data model for event records extracted from news articles.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class EventRecord:
    """
    Represents an event extracted from a news article.
    
    Attributes:
        event_type: Type of event (e.g., "PRZESTĘPSTWO", "WYPADEK", etc.)
        who: The subject/perpetrator of the event (WHO)
        what: The object/target of the event (WHAT)
        trigger: The verb/action that triggered the event
        where: Location of the event (optional)
        when: Time of the event (optional)
        confidence: Confidence score of the extraction (0-1)
        raw_sentence: Original sentence from which the event was extracted
    """
    event_type: str
    who: str
    what: str
    trigger: str
    where: Optional[str] = None
    when: Optional[str] = None
    confidence: float = 1.0
    raw_sentence: str = ""
    
    def __str__(self) -> str:
        """Human-readable representation of the event"""
        result = [
            f"Typ zdarzenia: {self.event_type}",
            f"KTO: {self.who}",
            f"CO: {self.what}",
            f"Trigger: {self.trigger}"
        ]
        
        if self.where:
            result.append(f"GDZIE: {self.where}")
        if self.when:
            result.append(f"KIEDY: {self.when}")
        if self.confidence < 1.0:
            result.append(f"Pewność: {self.confidence:.2f}")
            
        return "\n".join(result)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "event_type": self.event_type,
            "who": self.who,
            "what": self.what,
            "trigger": self.trigger,
            "where": self.where,
            "when": self.when,
            "confidence": self.confidence,
            "raw_sentence": self.raw_sentence
        }
