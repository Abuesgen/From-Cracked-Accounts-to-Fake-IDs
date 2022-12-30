"""
Module for plotting active times of an user
"""

# pylint: disable=missing-class-docstring,too-few-public-methods,invalid-name,missing-function-docstring,too-many-locals
from datetime import datetime, timedelta
from importlib.resources import path
from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import profiles
from profile_extraction.profile_creation.profile import (
    ProfileCollection,
    SummaryComponent,
)

pio.kaleido.scope.mathjax = None


def timestamps_to_times(post_times):
    times = [(datetime.strptime(date.time().strftime("%H:%M:%S"), "%H:%M:%S"),) for date in post_times]
    times.sort()
    return pd.DataFrame.from_records(times, columns=["post_time"])


def timestamps_to_post_dates(post_times):
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
    df = pd.DataFrame.from_records(list_dates, columns=["date", "count"])
    return df


MESSAGE_COUNT = "Message count"


def add_postimes(summary):
    df_dates = timestamps_to_post_dates(summary.post_times)
    line_plot = go.Figure(go.Scatter(x=df_dates["date"], y=df_dates["count"], mode="lines+markers"))
    line_plot.update_layout(
        title="Post history", xaxis_title="Date", yaxis_title=MESSAGE_COUNT, xaxis_tickformat="%d.%m.%Y"
    )

    df_times = timestamps_to_times(summary.post_times)

    time_from = datetime.strptime("00:00:00", "%H:%M:%S")
    time_to = time_from + timedelta(days=1)
    hist_plot = px.histogram(df_times, x="post_time", range_x=[time_from, time_to])
    hist_plot.update_traces(
        xbins=dict(  # bins used for histogram
            start=time_from,
            end=time_to,
            size=1000 * 3600,
        )
    )
    hist_plot.update_layout(
        xaxis_title=None,
        xaxis_tickformat="%Hh",
        xaxis_dtick=1000 * 60 * 60 * 4,
        yaxis_title="Messages per timeslot",
        font_family="Times New Roman",
        title_font_family="Times New Roman",
        font=dict(family="Times New Roman", size=24, color="black"),
        autosize=True,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    hist_plot.write_image("user_activity_times.pdf")

    # line_plot.show()


if __name__ == "__main__":
    with path(profiles, "schwarzmarkt_d.json") as profile_path:
        dataset = ProfileCollection.parse_file(profile_path)

        max_ts = -1

        for profile in dataset:

            tmp_sc = [c for c in profile.components if isinstance(c, SummaryComponent)][0]
            if len(tmp_sc.post_times) > max_ts:

                max_summary = tmp_sc
                max_ts = len(tmp_sc.post_times)

    add_postimes(max_summary)
