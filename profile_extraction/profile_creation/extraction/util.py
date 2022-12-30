"""
This module contains helper functions for profile extraction
"""
from typing import Any, Dict, Iterable, List, Tuple

from profile_extraction.profile_creation.profile import Entity, ProfileComponent


def ent_distances(ent: Entity, others: Iterable[Entity]) -> List[Tuple[Entity, float]]:
    """
    Calculates all distances from one entity to an Iterable of other entities
    It uses a token based euclidean distance see function ent_distance (See Also :func:`ent_distance`).

    :param others: Iterable of other entities to compute the distance to
    :param ent: Reference entity to compute the distance to

    :returns: List of tuples containing the reference entity and the calculated distance sorted by distance to ent
    """

    distances = [(other, ent_distance(ent, other)) for other in others]
    distances.sort(key=lambda item: item[1])
    return distances


def ent_distance(ent_one: Entity, ent_two: Entity) -> float:
    """
    Calculates the token distance from one entity to another.

    :param ent_one: First entity
    :param ent_two: Second entity
    """
    pos_one = ent_one.end
    pos_two = ent_two.start
    return abs(pos_two - pos_one)


def dict_list_append(key: Any, value: Any, add_to: Dict[Any, List[Any]]):
    """
    This function adds a value to dict list if the key exists.
    If the key does not exists the function creates the entry with a list of value
    Args:
      key: key of the dictionary
      value: value to add to the value list
      add_to: dictionary to add the key and value to
    """
    try:
        add_to[key].append(value)
    except KeyError:
        add_to[key] = [value]


def components_by_type(component_type, components: Iterable[ProfileComponent]):
    """
    Returns all ProfileComponents of a given type
    """
    for component in components:
        if isinstance(component, component_type):
            yield component
