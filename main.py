#!/usr/bin/env python
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import logging
import csv
import pytest
from inspect import currentframe


def log_lineno():
    cf = currentframe()
    lineno = cf.f_back.f_lineno
    logging.info(lineno)


logging.basicConfig(level="DEBUG")

Buchungstag = "Buchungstag"
Betrag = "Betrag"
Beguenstigter = "Beguenstigter"
Buchungstext = "Buchungstext"
Verwendungszweck = "Verwendungszweck"
Bankref = "Bankreferenz"
Ref_Transcode = "Zugehöriger Transaktionscode"
Transcode = "Transaktionscode"

# replace spaces spaces before !
data_map = {
    # "Valutadatum": "Wertstellung",
    # "Wertstellung": "Wertstellung",
    "Auftraggeber/Begünstigter": "Beguenstigter",
    "Beguenstigter/Zahlungspflichtiger": "Beguenstigter",
    "Betrag(EUR)": "Betrag",
    # "Kundenreferenz": "PPRef",
    # "Sammlerreferenz": "PPRef",
}
COLUMNS = pd.Index([Buchungstag, Betrag, Beguenstigter, Buchungstext, Verwendungszweck])
COLUMNS_EXTRA = pd.Index([])

PP_COLUMNS = pd.Index(
    [
        Bankref,
        Transcode,
        Ref_Transcode,
    ]
)
PP_COLUMNS_USER = pd.Index(
    [
        Transcode,
        "Name",
        "Artikelbezeichnung",
        "Rechnungsnummer",
        "Empfänger E-Mail-Adresse",
        "Hinweis",
    ]
)

assert COLUMNS.intersection(PP_COLUMNS).empty


def find_header(file, search: list, how="all", encoding=None):
    """how= 'all' or 'any'"""
    with open(file, "r", encoding=encoding) as f:
        i = -1
        for line in f:
            if len(line) <= 4:
                continue
            i += 1
            for sub in search:
                if how == "any" and sub in line:
                    return i
                if how == "all" and sub not in line:
                    break
                return i
    return None


def read_DKB(path) -> pd.DataFrame:
    encoding = "latin"
    header = find_header(
        path, ["Buchungstag", "Verwendungszweck"], how="all", encoding=encoding
    )
    df = pd.read_csv(
        path,
        sep=";",
        header=header,
        encoding=encoding,
        parse_dates=True,
        dayfirst=True,
    )
    df.attrs["name"] = "DKB"
    return df


def read_X(path) -> pd.DataFrame:
    # https://www.paypal.com/reports/dlog
    #   -> Berichtsfelder anpassen -> alle
    encoding = "latin"
    header = find_header(
        path, ["Buchungstag", "Verwendungszweck"], how="all", encoding=encoding
    )
    df = pd.read_csv(
        path,
        sep=";",
        header=header,
        encoding=encoding,
        parse_dates=True,
        dayfirst=True,
    )
    df.attrs["name"] = "X"
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.replace(" ", "")
    df = df.rename(data_map, axis=1).reindex(COLUMNS, axis=1)
    return df


def read_PayPal(path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=",",
        # index_col="Transaktionscode",
        encoding="utf8",
        parse_dates=True,
        dayfirst=True,
    )
    columns = df.columns.intersection(PP_COLUMNS.union(PP_COLUMNS_USER))
    df = df.reindex(columns, axis=1)
    assert Bankref in df.columns
    return df


def squeeze_paypal(df):
    # index_col="Transaktionscode",
    # todo sum netto brutto -> 0
    trans_id = df["Transaktionscode"] + df["Rechnungsnummer"]
    ref_id = df["Zugehöriger Transaktionscode"] + df["Rechnungsnummer"]
    return df


def join_append_paypal(data: pd.DataFrame, paypal: pd.DataFrame) -> pd.DataFrame:
    # bei Buchungstext==Lastschrift -> Verwendungszweck erste n nummern == Bankreferenz (pp_all)
    orig = data
    data = data.copy()

    # dd: direct debit
    dd = data[Buchungstext].isin(["Lastschrift", "FOLGELASTSCHRIFT"])
    pp = data[Beguenstigter].str.startswith("PayPal").fillna(False)
    mask = dd & pp
    # find VZ, that starts with a 13 digit (or more) number followed by a space
    mask &= data[Verwendungszweck].str.match("\d{12}\d+ .*")
    pp_ref = data[Verwendungszweck].str.split(" ", n=1, expand=True)[0]
    pp_ref[~mask] = np.nan
    data[Bankref] = pp_ref

    mask = paypal[Bankref].notna()
    to_add = paypal.loc[mask, [Bankref, Ref_Transcode]]
    df = pd.merge(data, to_add, on=Bankref, how="left")

    columns = PP_COLUMNS.union(PP_COLUMNS_USER).difference(pd.Index([Ref_Transcode]))
    to_add = paypal.loc[~mask, columns]
    df = pd.merge(df, to_add, left_on=Ref_Transcode, right_on=Transcode, how="left")

    # filter only requested columns
    df = df.reindex(orig.columns.union(PP_COLUMNS_USER, sort=False), axis=1)
    df.rename(lambda c: f"[PayPal] {c}" if c in PP_COLUMNS_USER else c, axis=1, inplace=True)
    return df


if __name__ == "__main__":
    assert pytest.main(["tests.py"]) == os.EX_OK
    data = read_DKB("Xenia/2022_DKB.csv")
    data = prepare_data(data)
    paypal = read_PayPal("Xenia/2022_PayPal_all.CSV")
    log_lineno()
    df = join_append_paypal(data, paypal)
    log_lineno()
    for c in df.columns:
        df[c] = df[c].str.slice(stop=50)
    df.to_csv("out.csv")
    # print("\n" + df.iloc[4:5, :].to_string())
