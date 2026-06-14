from pydantic import BaseModel


class KeyphraseItem(BaseModel):
    name: str
    variations: list[str]
    definition: str