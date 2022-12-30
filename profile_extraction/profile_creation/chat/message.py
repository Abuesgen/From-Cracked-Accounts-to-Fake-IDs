"""
A Message contains all important information of a Chat
"""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel

from profile_extraction.profile_creation.chat.user import TelegramUser


class Message(BaseModel):
    """
    This Class is used to parse a JsonChat using pydantics functions.
    It is used for easy converting to a "real" chat object.
    """

    id: int
    user: TelegramUser
    datetime: datetime
    message: Optional[str]
    reply_to: Optional[int]
    type: List[str]

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id
