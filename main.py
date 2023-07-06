#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/13 10:38
# @Author  : ZWP
# @Desc    : 
# @File    : main.py
import pandas as pd

df = pd.read_csv("./data/sales_train_validation.csv")


def fuck():
    print(df.describe())


if __name__ == '__main__':
    fuck()
