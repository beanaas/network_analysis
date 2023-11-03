#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSKS33 Hands-on session 4, load data

Erik G. Larsson 2020
"""

import snap
from gen_data import genmod10star
from gen_data import genLiveJournal

# -- load 10-star --
G, h = genmod10star()

# -- load LiveJournal --
G, h = genLiveJournal()
