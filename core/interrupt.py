import asyncio
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4


@dataclass
class InterruptController:
    generation_id: str = field(default_factory=lambda: str(uuid4()))
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    interrupted: bool = False

    def new_generation(self) -> str:
        self.cancel_all()
        self.generation_id = str(uuid4())
        self.cancel_event = asyncio.Event()
        self.interrupted = False
        return self.generation_id

    def cancel_all(self) -> None:
        self.interrupted = True
        self.cancel_event.set()

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def check(self, generation_id: str) -> bool:
        return generation_id == self.generation_id and not self.is_cancelled()