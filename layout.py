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
                    width=2,
                ),
                dbc.Col(
                    [
                        dbc.FormText(
                            dcc.Markdown(
                                "Download CSV from your Bank and upload here.",
                                link_target="_blank",  # open in new tab/window
                            ),
                            color="secondary",
                        ),
                    ],
                    width=8,
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
                    width=2,
                ),
                dbc.Col(
                    [
                        dbc.FormText(
                            dcc.Markdown(
                                "Login in Paypal. Then "
                                "1 click [https://www.paypal.com/reports/dlog](https://www.paypal.com/reports/dlog). "
                                "2 click Berichtsfelder anpassen -> select \[Transaktionsdetails, Warenkorbdetails, Payflow-Details\] -> (speichern). "
                                "3 select Datumsbereich, click Bericht erstellen. "
                                "4 repeatedly Aktualisieren until ready. "
                                "5 download CSV from list. "
                                "6.upload here ",
                                link_target="_blank",  # open in new tab/window
                            ),
                            color="secondary",
                        ),
                    ],
                    width=8,
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
