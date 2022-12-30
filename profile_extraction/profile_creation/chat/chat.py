"""
This module represents a Chat with all its messages.
It is used for every chat related operation like getting messages.
"""
from datetime import datetime
from typing import Iterable, List, Set

from pydantic import BaseModel, validator

from profile_extraction.profile_creation.chat.message import Message
from profile_extraction.profile_creation.chat.user import TelegramUser


class ChatInfo(BaseModel):
    """
    This class contains chat export metainformation such as export date message count and source
    """

    date: datetime
    message_count: int
    message_count_limit: int
    source: str


class Chat(BaseModel):
    """
    This Class is used to parse a JsonChat using pydantics functions.
    It is used for easy converting to a "real" chat object.
    """

    info: ChatInfo
    messages: List[Message]

    def __len__(self):
        return len(self.messages)

    @validator("messages")
    def validate_message_count(cls, messages, values):  # pylint: disable=no-self-argument
        """
        Checks whether given message count is the same as actual given messages.
        Self is named cls in this context because of pydantic. Otherwise pydantic
        will raise an Error.
        """
        if "info" in values and len(messages) != values["info"].message_count:
            raise ValueError(
                f"Actual message count does not match given message count. "
                f"Expected: {values['info'].message_count} Actual: {len(messages)}"
            )

        return messages

    @property
    def users(self) -> Set[TelegramUser]:
        """
        Returns all users in this chat
        """
        return {message.user for message in self.messages}

    def get_messages_for_user(self, user: TelegramUser) -> Iterable[Message]:
        """
        Returns all messages from a given user
        """
        for message in self.messages:
            if message.user == user:
                yield message
