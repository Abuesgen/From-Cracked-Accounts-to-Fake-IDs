"""
This modules contains a base User.
No matter what chat applications you use, there is always a way to
identify users.
"""
from typing import Optional

from pydantic import BaseModel


class TelegramUser(BaseModel):
    """This class represents a Chat user with all possible metadata Telegram has to offer"""

    id: str
    first_name: Optional[str]
    last_name: Optional[str]
    username: Optional[str]
    phone: Optional[str]

    def __hash__(self):
        """
        The id should be sufficient to identify a user in every case
        """
        return hash(self.id)

    def __eq__(self, other):
        """
        Checks whether two given TelegramUser are equal.
        Only if all fields are equal True is returned.
        """
        if not isinstance(other, TelegramUser):
            return False
        return (
            other.id == self.id
            and self.first_name == other.first_name
            and self.last_name == other.last_name
            and self.username == other.username
            and self.phone == other.phone
        )
