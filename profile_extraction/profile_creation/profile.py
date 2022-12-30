"""
This module contains the Profile class and its profile components
"""
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Set, Union

from flair.data import Span
from pydantic import BaseModel, Extra

from profile_extraction.profile_creation.chat.message import Message
from profile_extraction.profile_creation.chat.user import TelegramUser


class Entity(BaseModel):
    """
    This class represents a Named Entity
    """

    label: str
    text: str
    start: int
    end: int

    @classmethod
    def from_span(cls, span: Span):
        """
        Converts a flair span to an Entity object
        """
        return cls(label=span.labels[0].value, text=span.text, start=span.start_pos, end=span.end_pos)

    def __hash__(self):
        return hash((self.label, self.text, self.start, self.end))

    def __eq__(self, other):
        return (self.label, self.text, self.start, self.end) == (other.label, other.text, other.start, other.end)


class Relation(BaseModel):
    """
    This class represents a Relation between entities
    """

    head: Entity
    tail: Entity
    label: str


class BaseComponent(BaseModel):
    """
    This class represents a generic ProfileComponent
    """

    message: Message


class Product(BaseModel):
    """
    This class represents a Product
    """

    product: str
    price: Set[str] = set()
    message_ids: Set[int] = set()

    class Config:  # pylint: disable=too-few-public-methods
        """
        Pydantic extra configuration
        """

        extra = Extra.forbid


class NamedEntityComponent(BaseComponent):
    """
    This class represents a NamedEntityComponent
    """

    name: str = "NamedEntityComponent"
    entity: Entity

    class Config:  # pylint: disable=too-few-public-methods
        """
        Pydantic extra configuration
        """

        extra = Extra.forbid


class RelationComponent(BaseComponent):
    """
    Profile component describing an extracted relation
    """

    name: str = "RelationComponent"
    relation: Relation

    class Config:  # pylint: disable=too-few-public-methods
        """
        Pydantic extra configuration
        """

        extra = Extra.forbid


class ProductComponent(BaseComponent):
    """
    This class represents a Product with price
    """

    name: str = "ProductComponent"
    product: Entity
    price: Optional[Entity] = None

    class Config:  # pylint: disable=too-few-public-methods
        """
        Pydantic extra configuration
        """

        extra = Extra.forbid


class LocationComponent(BaseComponent):
    """
    This class represents a location
    """

    name: str = "LocationComponent"
    entity: Entity

    class Config:  # pylint: disable=too-few-public-methods
        """
        Pydantic extra configuration
        """

        extra = Extra.forbid


class PersonComponent(BaseComponent):
    """
    This class represents a mentioned person or user
    """

    name: str = "PersonComponent"
    entity: Entity

    class Config:  # pylint: disable=too-few-public-methods
        """
        Pydantic extra configuration
        """

        extra = Extra.forbid


class PaymentMethodComponent(BaseComponent):
    """
    This class represents a mentioned PaymentMethod
    """

    name: str = "PaymentMethodComponent"
    entity: Entity

    class Config:  # pylint: disable=too-few-public-methods
        """
        Pydantic extra configuration
        """

        extra = Extra.forbid


class SummaryComponent(BaseModel):
    """
    This class provides a short profile summary of all products, locations,
    persons payment_methods and message timestamps
    """

    name: str = "SummaryComponent"
    products: List[Product] = []
    locations: Dict[str, List[int]] = {}
    persons: Dict[str, List[int]] = {}
    payment_methods: Dict[str, List[int]] = {}
    post_times: List[datetime] = []

    class Config:  # pylint: disable=too-few-public-methods
        """
        Pydantic extra configuration
        """

        extra = Extra.forbid


ProfileComponent = Union[
    SummaryComponent,
    NamedEntityComponent,
    RelationComponent,
    ProductComponent,
    LocationComponent,
    PersonComponent,
    PaymentMethodComponent,
]


class Profile(BaseModel):
    """
    This class represents a single user profile.
    """

    user: TelegramUser
    components: List[ProfileComponent] = []

    def add(self, component: ProfileComponent):
        """
        Adds a ProfileComponent to the Profile
        """
        self.components.append(component)

    def extend(self, components: Iterable[ProfileComponent]):
        """
        Adds several ProfileComponents to the Profile
        """
        self.components.extend(components)

    def __hash__(self):
        return hash(self.user)

    def __eq__(self, other):
        if not isinstance(other, Profile):
            return False

        return self.user == other.user


class ProfileCollection(BaseModel):
    """
    This Collection represents a list of Profiles
    """

    profiles: List[Profile] = []

    def __len__(self):
        return len(self.profiles)

    def __iter__(self):
        return iter(self.profiles)

    def add(self, profile: Profile):
        """
        Adds a profile to the collection
        """
        self.profiles.append(profile)

    def extend(self, profiles: Iterable[Profile]):
        """
        Adds several profiles to the collection
        """
        self.profiles.extend(profiles)

    def __getitem__(self, item):
        if isinstance(item, str):
            for profile in self.profiles:
                if profile.user.id == item:
                    return profile
        else:
            for profile in self.profiles:
                if profile.user == item:
                    return profile

        raise KeyError(f'The user "{item}" has no profile in this collection.')
