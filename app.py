#!/usr/bin/env python

import dash
import os
from layout import layout


app = dash.Dash(
    name=__name__,
    title="Calconey",
    routes_pathname_prefix=os.getenv("APP_URL", None),
    requests_pathname_prefix=os.getenv("APP_URL", None),
    suppress_callback_exceptions=False,
)

app.layout = layout
