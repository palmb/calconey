#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
from __future__ import annotations
import base64
import io


import dash.exceptions
import pandas as pd
from dash import Input, Output, State, dcc

from app import app
from lib import (
    prepare_data,
    prepare_paypal,
    enrich,
    map_PayPal,
    read_PayPal,
    read_DKB,
    read_Bank,
)


def _get_trigger():
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    return ctx.triggered[0]["prop_id"].split(".")[0]


@app.callback(
    Output("result-content", "data"),
    Output("download-data", "disabled"),
    Input("download-data", "n_clicks"),
    Input("upload-data", "filename"),
    Input("upload-paypal", "filename"),
    State("upload-data", "contents"),
    State("upload-paypal", "contents"),
    prevent_initial_call=True,
)
def upload_datafile(click, dfile, ppfile, data, paypal):
    if dfile is None or data is None or ppfile is None or paypal is None:
        return dash.no_update, True

    if _get_trigger() != "download-data":
        return dash.no_update, False

    data = decode(data, encoding="latin")
    paypal = decode(paypal, encoding="utf8")
    df = calc(data, paypal)

    # df.to_csv("out.csv", index=False, decimal=",", sep=";", date_format="%d.%m.%y")
    return (
        dcc.send_data_frame(
            df.to_csv,
            "result.CSV",
            index=False,
            decimal=",",
            sep=";",
            date_format="%d.%m.%y",
        ),
        False,
    )


def decode(data, encoding="utf8"):
    content_type, content_string = data.split(",")
    return io.StringIO(base64.b64decode(content_string).decode(encoding))


def calc(data, paypal):
    df = read_Bank(data)
    pp = read_PayPal(paypal)
    df = prepare_data(df)
    pp = prepare_paypal(pp)
    pp = enrich(df, pp)
    assert df.index.equals(pp.index)
    result = map_PayPal(df, pp)
    # result.to_csv("out.csv", index=False, decimal=",", sep=";", date_format="%d.%m.%y")
    return result
