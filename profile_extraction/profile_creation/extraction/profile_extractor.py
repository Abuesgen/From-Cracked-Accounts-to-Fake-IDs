"""
This module contains the logic for profile extraction.
"""
import logging
from typing import Iterable, List, Tuple

from flair.data import Sentence
from flair.models import RelationExtractor, SequenceTagger
from flair.tokenization import SegtokTokenizer

from profile_extraction.profile_creation.chat.chat import Chat
from profile_extraction.profile_creation.chat.message import Message
from profile_extraction.profile_creation.extraction.extractors import (
    ExtractorEnum,
    ExtractorFactory,
)
from profile_extraction.profile_creation.profile import (
    Profile,
    ProfileCollection,
    SummaryComponent,
)

log = logging.getLogger(__name__)


def extract_profiles(
    chat: Chat, sequence_tagger: SequenceTagger, relation_extractor: RelationExtractor, summary_only: bool
) -> ProfileCollection:
    """
    Extracts all profiles from a given chat


    :param summary_only: Only return profiles with summary
    :param chat: chat containing all messages
    :param sequence_tagger: client to send request to las-web

    :returns: A ProfileCollection containing one profile for every user
    """

    profile_collection = ProfileCollection()

    extraction_steps = [
        ExtractorEnum.NER_ALL,
        ExtractorEnum.RELATIONS,
        ExtractorEnum.PRODUCT,
        ExtractorEnum.LOCATION,
        ExtractorEnum.PERSON,
        ExtractorEnum.PAYMENT,
        ExtractorEnum.SUMMARY,
    ]

    log.info("%6d messages will be processed using las-web.", len(chat))
    for user in chat.users:
        log.info("Building profile for User %s...", user.id)
        profile = Profile(user=user)

        message_results = prepare_messages(chat.get_messages_for_user(user), sequence_tagger, relation_extractor)

        for step in extraction_steps:
            extractor = ExtractorFactory.get_instance(step)
            profile.extend(extractor(message_results, profile))

        if summary_only:
            for component in profile.components[:]:
                if not isinstance(component, SummaryComponent):
                    profile.components.remove(component)
        profile_collection.add(profile)

    return profile_collection


def prepare_messages(
    messages: Iterable[Message], sequence_tagger: SequenceTagger, relation_extractor: RelationExtractor
) -> List[Tuple[Message, Sentence]]:
    """
    This method prepares all given chat messages for processing.

    All messages are analyzed with the LAS service and stored in a tuple together with the original message.

    :param messages: Messages to analyze using sequence_tagger
    :param sequence_tagger: client for the language analytics service

    :returns: All analyzed chat messages with the analysis results
    """
    messages_to_analyze = [
        message for message in messages if len((message.message if message.message else "").strip()) > 0
    ]

    sentences = [Sentence(message.message, use_tokenizer=SegtokTokenizer()) for message in messages_to_analyze]
    entity_label_type = "ner"
    sequence_tagger.predict(sentences, mini_batch_size=4, label_name=entity_label_type)
    relation_extractor.entity_label_type = entity_label_type
    relation_extractor.predict(sentences, mini_batch_size=4, label_name="rel")

    return list(zip(messages_to_analyze, sentences))
