# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc

form = dbc.Form(
    [
        dbc.FormGroup(
            [
                dbc.Label("Bank.csv ", width=2),
                dbc.Col(
                    [
                        dcc.Upload(
                            dbc.Button("Upload File"),
                            id="upload-data",
                        ),
                    ],
                    width="auto",
                ),
            ],
            row=True,
            inline=True,
        ),
        dbc.FormGroup(
            [
                dbc.Label("PayPal.csv ", width=2),
                dbc.Col(
                    [
                        dcc.Upload(
                            dbc.Button("Upload File"),
                            id="upload-paypal",
                        ),
                    ],
                    width="auto",
                ),
            ],
            row=True,
            inline=True,
        ),
        dbc.FormGroup(
            [
                dbc.Label("Result.csv", width=2, html_for="column-chooser"),
                dbc.Col(
                    dbc.Button("Download File", id="download-data", disabled=True),
                    width="auto",
                ),
                dcc.Download(id="result-content"),
            ],
            row=True,
            inline=True,
        ),
    ]
)


card = dbc.Card(
    [
        dbc.CardHeader(html.B("Calconey")),
        dbc.CardBody([form]),
        dbc.CardFooter("A fnert app"),
    ]
)

layout0 = dbc.Container(
    [
        html.Br(),
        card,
    ]
)

layout = layout0
