"""
This module contains all classes and functions to read chat data from
different sources. These sources are e.g. json Telegram Chats.
"""
from enum import Enum
from typing import Callable, Dict, TextIO

from profile_extraction.profile_creation.chat.chat import Chat

# Define Chat reader callable for convenience
ChatReader = Callable[[TextIO], Chat]


# Possible DataTypes
class DataType(str, Enum):
    """
    Enum which lists all possible DataTypes currently only json is supported
    """

    JSON = "json"


class ChatReaderFactory:
    """
    ChatReaderFactory creates the correct reader for a given DataType.
    """

    class_dictionary: Dict[DataType, ChatReader] = {}

    @classmethod
    def register(cls, data_type: DataType) -> Callable[[ChatReader], ChatReader]:
        """
        Registers a new reader to the factory

        :param data_type: data_type to register the new reader to

        :returns: Callable which registers the chat reader
        """

        def inner_wrapper(reader: ChatReader) -> ChatReader:
            cls.class_dictionary[data_type] = reader
            return reader

        return inner_wrapper

    @classmethod
    def get_instance(cls, data_type: DataType) -> ChatReader:
        """
        Returns the reader instance for a given DataType

        :param data_type: data_type of the reader

        :returns: Reader with DataType data_type if the given type exists

        :raises: ValueError: if data_type is not known
        """
        try:
            reader = cls.class_dictionary[data_type]
        except KeyError as err:
            known_formats = [key.value for key, _ in cls.class_dictionary.items()]
            raise ValueError(f"Unknown Data format. Known formats are: {known_formats}") from err

        return reader


@ChatReaderFactory.register(DataType.JSON)
def read_json(text_stream: TextIO) -> Chat:
    """
    This function parses a json chat file to a chat object using pydantics parse_raw function
    Args:
      text_stream: input_stream containing the json formatted chat data

    Returns:
      A chat object containing the read chat data
    """
    parsed_chat: Chat = Chat.parse_raw(text_stream.read())
    return parsed_chat
