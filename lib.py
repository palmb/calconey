#!/usr/bin/env python
from __future__ import annotations

import os
import re
import warnings

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


logging.basicConfig(level="INFO")

# Bank
Buchungstag = "Buchungstag"
Betrag = "Betrag"
Beguenstigter = "Beguenstigter"
Buchungstext = "Buchungstext"
Verwendungszweck = "Verwendungszweck"
# Paypal
Datum_Uhrzeit = "Datum_Uhrzeit"
Netto = "Netto"
Ref_Transcode = "Zugehöriger Transaktionscode"
Transcode = "Transaktionscode"
Bankref = "Bankreferenz"

# replace spaces spaces before !
data_map = {
    # "Valutadatum": "Wertstellung",
    # "Wertstellung": "Wertstellung",
    "Auftraggeber/Begünstigter": "Beguenstigter",
    "Beguenstigter/Zahlungspflichtiger": "Beguenstigter",
    "Betrag(EUR)": "Betrag",
}
COLUMNS = pd.Index([Buchungstag, Buchungstext, Betrag, Beguenstigter, Verwendungszweck])

PP_COLUMNS = pd.Index(
    [
        "Datum_Uhrzeit",
        Bankref,
        Transcode,
        Ref_Transcode,
        "Auswirkung auf Guthaben",
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
        thousands=".",
        decimal=",",
        header=header,
        encoding=encoding,
        parse_dates=[0, 1],
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
        thousands=".",  # seems not to work
        decimal=",",  # seems not to work
        header=header,
        encoding=encoding,
        parse_dates=[1, 2],
        dayfirst=True,
    )
    df.attrs["name"] = "X"
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.replace(" ", "")
    df = df.rename(data_map, axis=1).reindex(COLUMNS, axis=1)
    if pd.api.types.is_string_dtype(df[Betrag]):
        df[Betrag] = df[Betrag].str.replace(",", ".").astype(float)
    return df


def read_PayPal(path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=",",
        thousands=".",
        decimal=",",  # seems not to work
        encoding="utf8",
        dtype={Transcode: str, Ref_Transcode: str, Bankref: str},
        parse_dates=[[0, 1]],
        dayfirst=True,
    )
    columns = df.columns.intersection(PP_COLUMNS.union(PP_COLUMNS_USER))
    assert Bankref in columns
    return df


def ensure_columns(obj, columns, name):
    columns = pd.Index(columns)
    diff = columns.difference(obj.columns)
    if not diff.empty:
        raise ValueError(f"'{name}' missing these columns: {diff}")


def index_from_PP_DateTime(dates: pd.Series, times: pd.Series, **kwargs) -> pd.Index:
    """kwargs ar passed to to_datetime."""
    return pd.to_datetime(dates + "T" + times, **kwargs)


def map_PayPal_by_Date(data: pd.DataFrame, paypal: pd.DataFrame) -> pd.DataFrame:
    ensure_columns(data, [Buchungstag, Beguenstigter, Betrag], "data")
    ensure_columns(
        paypal,
        [
            Datum_Uhrzeit,
            Netto,
            Ref_Transcode,
            Transcode,
        ],
        "paypal",
    )
    data = data.copy()
    paypal = paypal.copy()

    # prepare data
    # data[Verwendungszweck] = data[Verwendungszweck].str.lower()
    mask = data[Beguenstigter].str.contains("paypal", case=False, na=False)
    dd = data[mask].copy()
    dd["dt"] = pd.to_datetime(data.loc[mask, Buchungstag], dayfirst=True)

    # prepare paypal

    index = pd.Series(data=paypal.index, index=pd.Index(paypal["Datum_Uhrzeit"]))

    df = pd.DataFrame(index=dd.index)
    pp_mask = pd.Series(True, index=paypal.index)
    for row in dd.itertuples():
        chunk = paypal.loc[index.loc[row.dt - pd.Timedelta("14d") : row.dt]]

        match = chunk["Netto"] == -float(row.Betrag)
        chunk = chunk[match]

        transcodes = chunk[Transcode].dropna()
        ref_transcodes = chunk[Ref_Transcode].dropna()
        count = transcodes.count()

        if count == 1:
            idx = transcodes.index[0]
        elif count == 2:
            if transcodes.iloc[0] == ref_transcodes.iloc[1]:
                idx = transcodes.index[0]
            elif ref_transcodes.iloc[0] == transcodes.iloc[1]:
                idx = transcodes.index[1]
            else:
                logging.debug("\n" + chunk.to_string())
                continue
        else:
            logging.debug("\n" + chunk.to_string())
            continue

        df.loc[row.Index, Ref_Transcode] = paypal.loc[idx, Ref_Transcode]
        pp_mask[idx] = False

    return merge_pp_to_data(
        df.reindex(data.index), paypal.loc[pp_mask], Ref_Transcode, Transcode
    )


def merge_pp_to_data(data: pd.DataFrame, paypal: pd.DataFrame, dcol, ppcol):
    assert dcol in data and ppcol in paypal
    data = data[[dcol]]
    columns = PP_COLUMNS.union(PP_COLUMNS_USER).difference(pd.Index([dcol]))
    # ensure no nans exist
    to_add = paypal.loc[paypal[ppcol].notna(), columns]
    if not to_add[ppcol].is_unique:
        warnings.warn("right side of merge has non-unique values", stacklevel=2)
    df = pd.merge(data, to_add, left_on=dcol, right_on=ppcol, how="left")
    # filter only requested columns
    return df.reindex(pd.Index(PP_COLUMNS_USER), axis=1)


def map_PayPal_by_Bankreferenz(
    data: pd.DataFrame, paypal: pd.DataFrame
) -> pd.DataFrame:
    # 1. data.Buchungstext=="Lastschrift" & data.Verwendungszweck=="PayPal..."
    # 2. extract and store first 13 (or more digits) from data.Beguenstigter as data.Bankreferenz
    # 3. join data and paypal on data.Bankreferenz==paypal.Bankreferenz
    # 4. return df with only paypal columns
    data = data.copy()
    ensure_columns(data, [Buchungstext, Beguenstigter, Verwendungszweck], "data")
    ensure_columns(paypal, [Bankref, Ref_Transcode, Transcode], "paypal")

    # dd: direct debit
    dd = data[Buchungstext].isin(["Lastschrift", "FOLGELASTSCHRIFT"])
    pp = data[Beguenstigter].str.contains("PayPal").fillna(False)
    mask = dd & pp
    # find VZ, that starts with a 13 digit (or more) number followed by a space
    mask &= data[Verwendungszweck].str.match(r"\d{12}\d+ .*")
    pp_ref = data[Verwendungszweck].str.split(" ", n=1, expand=True)[0]
    pp_ref[~mask] = np.nan
    data[Bankref] = pp_ref

    mask = paypal[Bankref].notna()
    to_add = paypal.loc[mask, [Bankref, Ref_Transcode]]
    df = pd.merge(data, to_add, on=Bankref, how="left")

    return merge_pp_to_data(df, paypal.loc[~mask], Ref_Transcode, Transcode)


def map_PayPal(data: pd.DataFrame, paypal: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    name = paypal["Name"]
    sel = name.notna()
    data.loc[sel, Beguenstigter] = name[sel]

    artikel = paypal["Artikelbezeichnung"]
    hint = paypal["Hinweis"]
    sel = artikel.notna() & hint.notna()
    artikel[sel] += ", " + hint[sel]
    sel = artikel.notna()
    data.loc[sel, Verwendungszweck] = artikel[sel]

    data["[Paypal] Transaktionscode"] = paypal[Transcode]
    data["[Paypal] Rechnungsnummer"] = paypal["Rechnungsnummer"]
    data["[Paypal] Empfänger E-Mail-Adresse"] = paypal["Empfänger E-Mail-Adresse"]

    return data


def add_PayPal(data, paypal, prefix="[PayPal] ", suffix=""):
    paypal = paypal.copy()
    paypal.columns = prefix + paypal.columns + suffix
    return data.join(paypal)


def prepare_paypal(df):
    # ignore Memos as they are PayPal internal
    mask = df["Auswirkung auf Guthaben"] == "Memo"
    df = df[~mask]
    return df


def enrich(data: pd.DataFrame, paypal: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    paypal = paypal.copy()
    # inprc = paypal["Status"] == "Ausstehend"
    # done = ~inprc
    # dupes = paypal.loc[inprc, Datum_Uhrzeit].isin(paypal.loc[done, Datum_Uhrzeit])
    # dupes = dupes.reindex(paypal.index, fill_value=False)
    # paypal = paypal[~dupes]
    paypal["index"] = paypal.index
    _paypal = {
        "lower": paypal[paypal[Netto] < 0].set_index(Datum_Uhrzeit, drop=True),
        "upper": paypal[paypal[Netto] >= 0].set_index(Datum_Uhrzeit, drop=True),
        Bankref: paypal.set_index(Bankref, drop=True),
        Transcode: paypal.set_index(Transcode, drop=True),
    }

    def reffind_ppIndex(br):
        return _paypal[Bankref]["index"].get(br)

    def transfind_ppIndex(tc):
        if tc is None or pd.isna(tc):
            return None
        return _paypal[Transcode]["index"].get(tc)

    def is_paypal(idx):
        # return (
        #     "paypal" in str(data.loc[idx, Beguenstigter]).lower()
        #     or "paypal" in str(data.loc[idx, Verwendungszweck]).lower()
        # )
        return data.loc[idx, [Beguenstigter, Verwendungszweck]].str.contains(
            "paypal", case=False, na=False, regex=False
        ).any()

    rx_ref = re.compile(r"(\d{12}\d+) .*")

    def get_bankref(idx_) -> str | None:
        match = rx_ref.match(data.loc[idx_, Verwendungszweck])
        if match is not None:
            match = match.group(1)
        return match

    def datefind_ppIndex(date, value, delta="14d"):
        # Note this drops the index !
        if value > 0:
            side = "lower"
            sl = slice(date - pd.Timedelta(delta), date)
        else:
            side = "upper"
            sl = slice(date - pd.Timedelta(delta), date)
        chunk = _paypal[side].loc[sl, [Netto, "index"]]
        match = chunk[chunk[Netto] == -value]
        if match.empty:
            return None
        # We take the first of multiple values and
        # ignore the other. The other maybe are used
        # wth the next call, and an exact match for
        # the very same value of 'value' (might) not
        # be needed. This is the back-draw of this
        # method, but who cares.
        return match.iloc[0, 1]

    def drop_ppRow(i):
        br = paypal.loc[i, Bankref]
        _paypal[Bankref].drop(br, axis=0, inplace=True, errors='ignore')
        tc = paypal.loc[i, Transcode]
        _paypal[Transcode].drop(tc, axis=0, inplace=True, errors='ignore')
        dt = paypal.loc[i, Datum_Uhrzeit]
        _paypal['lower'].drop(dt, axis=0, inplace=True, errors='ignore')
        _paypal['upper'].drop(dt, axis=0, inplace=True, errors='ignore')
        paypal.drop(i, axis=0, inplace=True, errors='raise')  # raise !

    result = pd.DataFrame(index=data.index, columns=paypal.columns)
    for idx, row in data.iterrows():
        row: pd.Series
        if is_paypal(idx):
            bankref = get_bankref(idx)
            if bankref is None:  # manual search
                ppidx = datefind_ppIndex(row[Buchungstag], row[Betrag])
            else:
                ppidx = reffind_ppIndex(bankref)

            if ppidx is not None:
                # We use the data of the reference transaction code
                # instead of the original transaction code, because
                # it holds the relevant information.
                ppidx2 = transfind_ppIndex(paypal.loc[ppidx, Ref_Transcode])
                if ppidx2 is not None:
                    drop_ppRow(ppidx)
                    ppidx = ppidx2
                result.loc[idx] = paypal.loc[ppidx]
                drop_ppRow(ppidx)
            # overwrite/append Verwendungszweck / Begünstigter etc

        # map data to categories
    paypal = paypal.loc[:, :"Status"]
    return result
