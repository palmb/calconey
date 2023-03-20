#!/usr/bin/env python
from __future__ import annotations

import pandas as pd
import logging
import csv

logging.basicConfig(level="DEBUG")


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
    return pd.read_csv(
        path,
        sep=";",
        header=header,
        encoding=encoding,
        parse_dates=True,
        dayfirst=True,
    )


def read_X(path) -> pd.DataFrame:
    # https://www.paypal.com/reports/statements/custom
    encoding = "latin"
    header = find_header(
        path, ["Buchungstag", "Verwendungszweck"], how="all", encoding=encoding
    )
    return pd.read_csv(
        path,
        sep=";",
        header=header,
        encoding=encoding,
        parse_dates=True,
        dayfirst=True,
    )


def read_PayPal(path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=",",
        # index_col="Transaktionscode",
        encoding="utf8",
        parse_dates=True,
        dayfirst=True,
    )


def test():
    df = read_X("Xenia/EÜR 2022 neu.csv")
    df = read_DKB("Xenia/2022_DKB.csv")
    df = read_PayPal("Xenia/2022_PayPal.CSV")

# todo: PayPal
#   parse
#    data.sammlerreferenz
#   join mit
#    paypal(1).transaktionscode
#   join mit paypal(2).zugehöriger transaktionscode
#    paypal(2) is die buchung

def squeeze_paypal(df):
    # index_col="Transaktionscode",
    # todo sum netto brutto -> 0
    left = df["Transaktionscode"]
    right = df["Zugehöriger Transaktionscode"]

def join_append_paypal(data: pd.DataFrame, paypal: pd.DataFrame) -> pd.DataFrame:

    df = data.join(paypal, on="Foo", how='left', rsuffix='pp_')
    return df

if __name__ == "__main__":
    test()
    data = read_DKB("Xenia/2022_DKB.csv")
    paypal = read_PayPal("Xenia/2022_PayPal.CSV")
    df = join_append_paypal(data, paypal)
    # print("\n" + df.iloc[4:5, :].to_string())
