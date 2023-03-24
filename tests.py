#!/usr/bin/env python
from __future__ import annotations

import pytest
from lib import read_X, read_DKB, read_PayPal, prepare_data, COLUMNS


def test_read():
    df = read_DKB("Xenia/2022_DKB.csv")
    df = read_PayPal("Xenia/2022_PayPal_all.CSV")
    df = read_X("Xenia/EÜR 2022 neu.csv")


@pytest.mark.parametrize(
    "df",
    [
        read_DKB("Xenia/2022_DKB.csv"),
        read_X("Xenia/EÜR 2022 neu.csv"),
    ],
)
def test_prepare(df):
    data = prepare_data(df)
    assert data.columns.equals(COLUMNS)
    for c in data.columns:
        print(c)
        assert data[c].notna().any()


# todo: PayPal
#   parse
#    data.sammlerreferenz
#   join mit
#    paypal(1).transaktionscode
#   join mit paypal(2).zugehöriger transaktionscode
#    paypal(2) is die buchung


