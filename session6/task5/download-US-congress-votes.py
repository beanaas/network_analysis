#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSKS33 hands-on session 6 task 5

Download U.S. congress votes and save to disk

Erik G. Larsson, 2020
"""

import pandas as pd

for i in range(699):
    url = "https://www.govtrack.us/congress/votes/116-2019/h"+str(i+3)+"/export/csv"
    data=pd.read_csv(url,skiprows=1)
    data.to_pickle('./vote-116-2019-H'+str(i))
