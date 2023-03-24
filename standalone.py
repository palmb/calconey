#!/usr/bin/env python
from __future__ import annotations

import pytest
from lib import (
    read_DKB,
    read_PayPal,
    read_X,
    prepare_data,
    prepare_paypal,
    map_PayPal,
    enrich,
    add_PayPal,
)
import os
import pandas as pd


if __name__ == "__main__":
    pd.options.display.max_columns = 10
    pd.options.display.width = 150
    assert pytest.main(["tests.py"]) == os.EX_OK
    df = read_DKB("Xenia/2022_DKB.csv")
    df = prepare_data(df)
    paypal = read_PayPal("Xenia/2022_PayPal_all.CSV")
    pp = prepare_paypal(paypal)
    pp = enrich(df, pp)
    assert df.index.equals(pp.index)
    df = map_PayPal(df, pp)
    df.to_csv("out.csv", index=False, decimal=",", sep=";", date_format="%d.%m.%y")
