"""
This module contains commands to generate a html representation of chatprofile data
"""
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import dominate
import pandas as pd
from dominate.tags import (
    a,
    b,
    br,
    div,
    h1,
    h2,
    li,
    link,
    mark,
    meta,
    script,
    span,
    style,
    table,
    tbody,
    td,
    th,
    thead,
    tr,
    ul,
)
from dominate.util import raw

from profile_extraction.profile_creation.chat.message import Message
from profile_extraction.profile_creation.profile import (
    Entity,
    NamedEntityComponent,
    Profile,
    SummaryComponent,
)

MESSAGE_COUNT = "Message count"
COLORS = {
    "LOC": "#5C8AA8",
    "PAYM": "#FB2050",
    "PER": "#CC8033",
    "MONEY": "#8CB500",
    "PROD": "#9966CC",
    "CRIT": "#FCDC00",
}


def create_html(profile: Profile):
    """
    Creates a HTML representation from a given profile
    :param nlp: dictionary containing string to flair sentences mapping for NE rendering
    :param profile: Profile to render
    :return: String containing a rendered HTML
    """
    user = profile.user
    summary = None
    for component in profile.components:
        if isinstance(component, SummaryComponent):
            summary = component

    html_title = f"Profile for User '{user.username} ({user.id})'"
    doc = dominate.document(title=html_title)

    with doc.head:
        meta(charset="utf-8")
        meta(name="viewport", content="width=device-width, initial-scale=1")
        link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css")
        script(type="text/javascript", src="https://code.jquery.com/jquery-3.5.1.slim.min.js")
        script(
            type="text/javascript", src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/js/bootstrap.bundle.min.js"
        )
        script(src="https://twemoji.maxcdn.com/v/latest/twemoji.min.js", crossorigin="anonymous")
        with style():
            raw("img.emoji {height: 1em;width: 1em;margin: 0 .05em 0 .1em;vertical-align: -0.1em;}")

    with doc:
        with div(cls="jumbotron text-center"):
            h1(html_title)

        with div(cls="container"):
            add_user_info(user)

            add_products(profile, summary)

            add_payments(summary)

            add_locations(summary)

            add_persons(summary)

        script().add(raw("twemoji.parse(document.body);"))

    return doc.render(pretty=False)


def timestamps_to_times(post_times):
    """
    Creates a pandas Dataframe containing the time from timestamps
    :param post_times: timestamps to extract the time from
    :return: pandas DataFrame containing times
    """
    times = [(datetime.strptime(date.time().strftime("%H:%M:%S"), "%H:%M:%S"),) for date in post_times]
    times.sort()
    return pd.DataFrame.from_records(times, columns=["post_time"])


def timestamps_to_post_dates(post_times):
    """
    Creates a dataframe for a post histogram showing when a user posted most messages
    :param post_times: message timestamps
    :return:  a dataframe containing date and messages per date information
    """
    if len(post_times) == 0:
        return pd.DataFrame(columns=["date", "count"])
    dates: Dict[datetime, int] = {}
    for post_time in post_times:
        try:
            dates[post_time.date()] += 1
        except KeyError:
            dates[post_time.date()] = 1

    min_date: datetime = min(post_times).date()
    max_date: datetime = max(post_times).date()
    while min_date <= max_date:
        if min_date not in dates.keys():  # pylint: disable=consider-iterating-dictionary
            dates[min_date] = 0
        min_date += timedelta(days=1)

    list_dates = list(dates.items())
    list_dates.sort(key=lambda item: item[0])
    data_frame = pd.DataFrame.from_records(list_dates, columns=["date", "count"])
    return data_frame


def add_persons(summary):
    """
    Adds mentioned users to the HTML
    :param summary: SummaryComponent containing mentioned users
    """
    with div(id="persons", cls="row"):
        with div(cls="col-sm-12"):
            h2("Persons")
            dict_table(["Person", MESSAGE_COUNT], summary.persons)


def add_locations(summary):
    """
    Adds mentioned locations to the HTML
    :param summary:  SummaryComponent containing mentioned locations
    """
    with div(id="locations", cls="row"):
        with div(cls="col-sm-12"):
            h2("Locations")
            dict_table(["Location", MESSAGE_COUNT], summary.locations)


def add_payments(summary):
    """
    Adds payment methods to the HTML
    :param summary: SummaryComponent containing mentioned payment methods
    """
    with div(id="payment", cls="row"):
        with div(cls="col-sm-12"):
            h2("Payment Methods")
            dict_table(["Method", MESSAGE_COUNT], summary.payment_methods)


def add_products(profile, summary):
    """
    Adds products to the HTML
    :param nlp: Mapping from message str to analyzed Sentences containing NE information
    :param profile: Profile to get the products from
    :param summary: SummaryComponent containing all mentioned products
    """
    with div(id="products", cls="row"):
        with div(cls="col-sm-12", id="product_accordion"):
            h2("Products")
            product_id = 0
            for product in summary.products:
                add_product(product, profile, product_id)
                product_id += 1


def add_user_info(user):
    """
    Adds a user Metadata table to the generated HTML
    :param user: User to render
    """
    with div(id="user", cls="row"):
        with div(cls="col-sm-12"):
            h2("User")
            with table(cls="table table-striped"):
                with tbody():
                    with tr():
                        td().add(b("ID"))
                        td(ensure_not_none(user.id))

                    with tr():
                        td().add(b("Username"))
                        td(ensure_not_none(user.username))

                    with tr():
                        td().add(b("First name"))
                        td(ensure_not_none(user.first_name))

                    with tr():
                        td().add(b("Last Name"))
                        td(ensure_not_none(user.last_name))

                    with tr():
                        td().add(b("Telephone"))
                        td(ensure_not_none(user.phone))


def add_product(product, profile, prod_id):
    """
    Adds a product accordion to the generated HTML
    """
    with div(cls="card"):
        with div(cls="card-header row"):
            with div(cls="col-sm-12").add(table(cls="table")):
                with thead().add(tr()):
                    th("Product")
                    th("Prices")
                    th(MESSAGE_COUNT)
                with tbody().add(tr()):
                    td(product.product, width="50%")
                    with td(width="30%").add(ul()):
                        for price in product.price:
                            li(price)
                    td(len(product.message_ids))

            with div(cls="col-sm-12"):
                a("Show Messages", cls="btn btn-primary", href=f"#prod_{prod_id}", data_toggle="collapse")

        with div(id=f"prod_{prod_id}", cls="", data_parent="#product_accordion"):
            with div(cls="card-body"):
                add_messages(profile, product.message_ids)


def get_message(message_id: int, profile: Profile) -> Optional[Message]:
    """
    Helper function to get a message by its ID
    :param message_id: ID to search for
    :param profile: Profile containing the message
    :return: The relevant message or None
    """
    for component in profile.components:
        try:
            if component.message.id == message_id:  # type: ignore
                return component.message  # type: ignore
        except KeyError:
            pass
    return None


def add_messages(profile: Profile, message_ids: Iterable[int]):
    """
    Adds a message table containing several messages to the HTML
    :param profile: Profile to get the messages from
    :param message_ids: Message IDs to render
    :param nlp: dict containing str to Sentence mapping for NE rendering
    """
    all_messages: List[Optional[Message]] = [get_message(message_id, profile) for message_id in message_ids]
    messages: List[Message] = [message for message in all_messages if message is not None]

    message_dict: Dict[str, List[int]] = {ensure_not_none(message.message): [] for message in messages}
    for message in messages:
        message_dict[ensure_not_none(message.message)].append(message.id)

    with table(cls="table table-responsive table-striped", style="max-height: 50vh"):
        with thead().add(tr()):
            th("Message IDs")
            th("Message")

        with tbody():
            for message_str, ids in message_dict.items():
                add_message(message_str, ids, profile)


def get_components_for_message(
    message_id: int,
    profile: Profile,
):
    """
    Retrieves all components for a given message
    Args:
        message_id:
        profile:

    Returns:

    """
    components = [
        component
        for component in profile.components
        if not isinstance(component, SummaryComponent) and component.message.id == message_id
    ]
    return components


def add_message(message: str, ids: List[int], profile: Profile):
    """
    Adds a message with NEs to the generated HTML
    :param nlp: dict containing str to Sentence mapping for NE rendering
    :param message: message to render
    :param ids: ids corresponding to the same text
    """
    ids.sort()
    components = get_components_for_message(ids[0], profile)

    with tr():
        with td(width="20%"):
            with ul():
                for mid in ids:
                    li(mid)

        with td():
            with div(cls="entities", style="line-height: 2.5; direction: ltr"):
                all_entities: List[Entity] = [
                    entity.entity for entity in components if isinstance(entity, NamedEntityComponent)
                ]
                offset = 0
                for line in message.splitlines():
                    entities = [
                        entity
                        for entity in all_entities
                        if entity.start < offset + len(line) and entity.start >= offset
                    ]
                    if len(entities) > 0:
                        entities.sort(key=lambda x: x.start)

                        end_idx = 0

                        for entity in entities:
                            span(line[(end_idx - offset) : (entity.start - offset)])

                            tag = entity.label
                            with mark(
                                id=f"{ids[0]}-{entity.start}-{entity.end}-{entity.label}",
                                cls="entity",
                                style=f"background: {COLORS[tag]}; padding: 0.45em 0.6em; margin: 0 0.25em; "
                                "line-height: 1; border-radius: 0.35em;",
                            ) as marked_entity:
                                marked_entity.add(line[(entity.start - offset) : (entity.end - offset)])
                                with span(
                                    style="font-size: 0.8em; font-weight: bold; line-height: 1; "
                                    "border-radius: 0.35em; vertical-align: middle; "
                                    "margin-left: 0.5rem"
                                ) as sp_label:
                                    sp_label.add(tag)

                                end_idx = entity.end
                        span(line[end_idx:])
                    else:
                        span(line)
                    offset += len(line) + len(os.linesep)
                    br()


def ensure_not_none(data) -> str:
    """
    Ensures a string is not None. defaults to ""
    :param data: data to ensure not None
    :return: str representation of the object
    """
    return str(data) if data is not None else ""


def dict_table(headers: List[str], dictionary):
    """
    Creates a Table from a dictionary
    :param headers: headers for the table
    :param dictionary: dictionary containing table data
    """
    if len(headers) != 2:
        raise ValueError("Currently only two headers are supported by dict_table")

    data_tuple = list(zip(dictionary.keys(), dictionary.values()))
    tuple_table(headers, data_tuple)


def tuple_table(headers: List[str], data_tuples: List[Tuple[Any, Any]]):
    """
    Creates a table from tuples
    :param headers: table headers
    :param data_tuples: table data
    """
    with table(cls="table table-striped"):
        if headers is not None:
            with thead():
                with tr():
                    th(headers[0])
                    th(headers[1])
        with tbody():
            for tup in data_tuples:
                with tr():
                    for entry in tup:
                        if isinstance(entry, List):
                            td(len(entry))
                        else:
                            td(entry)


def create_profile_visualization(output_dir, profile):
    """
    Create a HTML visualization for a single profile
    :param nlp: mapping of str to processed sentences
    :param output_dir: dir to write the HTML file to
    :param profile: Profile to render
    """
    html = create_html(profile)
    html_file = Path(output_dir).joinpath(Path(f"{profile.user.id}.html"))
    with html_file.open("w", encoding="utf-8") as file:
        file.write(html)
