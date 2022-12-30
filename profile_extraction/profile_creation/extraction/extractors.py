"""
This module contains all extraction Components for Profiles.
"""
from enum import Enum
from typing import Callable, Dict, Iterable, List, Tuple

from flair.data import RelationLabel, Sentence

from profile_extraction.profile_creation.chat.message import Message
from profile_extraction.profile_creation.extraction.util import (
    components_by_type,
    dict_list_append,
    ent_distances,
)
from profile_extraction.profile_creation.profile import (
    Entity,
    LocationComponent,
    NamedEntityComponent,
    PaymentMethodComponent,
    PersonComponent,
    Product,
    ProductComponent,
    Profile,
    ProfileComponent,
    Relation,
    RelationComponent,
    SummaryComponent,
)

Extractor = Callable[[Iterable[Tuple[Message, Sentence]], Profile], List[ProfileComponent]]


class ExtractorEnum(str, Enum):
    """
    This enum contains all known extractors as names
    """

    RELATIONS = "RELATIONS"
    NER_ALL = "NER_ALL"
    PRODUCT = "PRODUCT"
    SUMMARY = "SUMMARY"
    LOCATION = "LOCATION"
    PERSON = "PERSON"
    PAYMENT = "PAYMENT"


class ExtractorFactory:
    """
    This factory creates the correct profile component extractors by their name (ExtractorEnum)
    """

    class_dictionary: Dict[ExtractorEnum, Extractor] = {}

    @classmethod
    def register(cls, extractor_type: ExtractorEnum) -> Callable[[Extractor], Extractor]:
        """
        Registers a new extractor

        :param extractor_type: Name of the extractor to register
        """

        def inner_wrapper(extractor: Extractor) -> Extractor:
            cls.class_dictionary[extractor_type] = extractor
            return extractor

        return inner_wrapper

    @classmethod
    def get_instance(cls, extractor_type: ExtractorEnum) -> Extractor:
        """
        Creates a new extractor of type extractor_type
        """
        try:
            extractor = cls.class_dictionary[extractor_type]
        except KeyError as err:
            known_formats = [key.value for key, _ in cls.class_dictionary.items()]
            raise ValueError(f"Unknown extractor format. Known extractors are: {known_formats}") from err

        return extractor


@ExtractorFactory.register(ExtractorEnum.NER_ALL)  # type: ignore
def named_entity_extractor(messages: Iterable[Tuple[Message, Sentence]], _: Profile) -> List[NamedEntityComponent]:
    """
    Extracts all Named Entities from an analyzed text and returns a list of all extracted Named Entities

    :returns: list of all Named Entities and their corresponding chat message
    """
    named_entities = []
    for message, sentence in messages:
        named_entities.extend(
            [
                NamedEntityComponent(message=message, entity=Entity.from_span(span))
                for span in sentence.get_spans("ner")
            ]
        )

    return named_entities


@ExtractorFactory.register(ExtractorEnum.RELATIONS)
def relation_extractor(messages: Iterable[Tuple[Message, Sentence]], _: Profile) -> List[RelationComponent]:
    """
    Extract all relations present in the messages of a user
    Args:
        messages: Tuple of messages and predicted sentences
        _: current profile, contains all data extracted up to this point

    Returns:
        A list of all extracted relations.
    """
    relations: List[RelationComponent] = []

    for message, sentence in messages:
        relations.extend(
            [
                RelationComponent(
                    message=message,
                    relation=Relation(
                        head=Entity.from_span(relation.head),
                        tail=Entity.from_span(relation.tail),
                        label=relation.value,
                    ),
                )
                for relation in sentence.get_labels("rel")
                if isinstance(relation, RelationLabel)
            ]
        )

    return relations


@ExtractorFactory.register(ExtractorEnum.LOCATION)  # type: ignore
def location_extractor(messages: Iterable[Tuple[Message, Sentence]], _: Profile) -> List[LocationComponent]:
    """
    This extractor extracts all locations from all messages a user has written

    :returns: List of extracted locations
    """
    locations: List[LocationComponent] = []
    for message, sentence in messages:
        locations.extend(
            [
                LocationComponent(message=message, entity=Entity.from_span(span))
                for span in sentence.get_spans("ner")
                if span.labels[0].value == "LOC"
            ]
        )

    return locations


@ExtractorFactory.register(ExtractorEnum.PERSON)  # type: ignore
def person_extractor(messages: Iterable[Tuple[Message, Sentence]], _: Profile) -> List[PersonComponent]:
    """
    This method extracts all mentioned person names in all messages a user has sent

    :param messages: a list of tuples of chat messages and their nlp results
    :param  _: ignored

    :returns: A List of PersonComponent containing all found persons.
    """
    persons: List[PersonComponent] = []
    for message, sentence in messages:
        persons.extend(
            [
                PersonComponent(message=message, entity=Entity.from_span(span))
                for span in sentence.get_spans("ner")
                if span.labels[0].value == "PER"
            ]
        )

    return persons


@ExtractorFactory.register(ExtractorEnum.PAYMENT)  # type: ignore
def payment_extractor(messages: Iterable[Tuple[Message, Sentence]], _: Profile) -> List[PaymentMethodComponent]:
    """
    This method extracts all mentioned person names in all messages a user has sent

    :param messages: a list of tuples of chat messages and their nlp results
    :param  _: ignored

    :returns: A List of PaymentMethodComponent containing all found payment_methods.
    """
    payment_methods: List[PaymentMethodComponent] = []
    for message, sentence in messages:
        payment_methods.extend(
            [
                PaymentMethodComponent(message=message, entity=Entity.from_span(span))
                for span in sentence.get_spans("ner")
                if span.labels[0].value == "PAYM"
            ]
        )

    return payment_methods


@ExtractorFactory.register(ExtractorEnum.PRODUCT)  # type: ignore
def product_extractor(messages: Iterable[Tuple[Message, Sentence]], _: Profile) -> List[ProductComponent]:
    """
    Extracts all Products and their Prices (If possible)
    """
    products: List[ProductComponent] = []

    for message, sentence in messages:
        prices = [Entity.from_span(span) for span in sentence.get_spans("ner") if span.labels[0].value == "MONEY"]
        prods = [Entity.from_span(span) for span in sentence.get_spans("ner") if span.labels[0].value == "PROD"]

        for prod in prods:
            product = ProductComponent(message=message, product=prod)
            distances = ent_distances(prod, prices)
            if len(distances) > 0:
                price = distances[0][0]
                product.price = price
            products.append(product)

    return products


@ExtractorFactory.register(ExtractorEnum.SUMMARY)  # type: ignore
def summary_extractor(messages: Iterable[Tuple[Message, Sentence]], profile: Profile) -> List[SummaryComponent]:
    """
    This method creates a summary of the whole user profile. It has to be placed at the end of a pipeline

    :param  messages: a list of tuples of chat messages and their nlp results
    :param  profile: Profile containing all collected data during the pipeline


    :returns: A List containing ONE SummaryComponent.
    """
    summary = SummaryComponent()

    # Add Products to summary
    product_dict: Dict[str, Product] = {}
    for component in components_by_type(ProductComponent, profile.components):
        product_key = component.product.text.lower()
        try:
            prod = product_dict[product_key]
        except KeyError:
            prod = Product(product=component.product.text)

        if component.price is not None:
            prod.price.add(component.price.text)
        prod.message_ids.add(component.message.id)

        product_dict[product_key] = prod
    summary.products.extend(product_dict.values())
    summary.products.sort(key=lambda item: item.product)

    # Add locations to summary
    for component in components_by_type(LocationComponent, profile.components):
        dict_list_append(component.entity.text, component.message.id, summary.locations)

    # Add Persons to summary
    for component in components_by_type(PersonComponent, profile.components):
        dict_list_append(component.entity.text, component.message.id, summary.persons)

    # Add Payment methods to summary
    for component in components_by_type(PaymentMethodComponent, profile.components):
        dict_list_append(component.entity.text, component.message.id, summary.payment_methods)

    # Add message timestamps to summary
    for message, _ in messages:
        if message.datetime not in summary.post_times:
            summary.post_times.append(message.datetime)
    summary.post_times.sort()

    return [summary]
