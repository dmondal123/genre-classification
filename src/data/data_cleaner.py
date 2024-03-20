import numpy as np
import distance

import os
import datetime
import sys 

import pandas as pd

def merge_near_string(what, df):
    """This function merges similar strings within a DataFrame column by replacing
    shorter strings with their longer counterparts if their Levenshtein distance is
    less than or equal to 1."""
    uq = df[what][df[what].notna()].unique()

    replace_list = {}
    for w1 in uq:
        if len(w1) < 5:
            continue
        if w1 in replace_list:
            continue
        for w2 in uq:
            if len(w2) < 5:
                continue
            if w1 == w2:
                continue
            if distance.levenshtein(w1,w2) > 1:
                continue
            if w2 in replace_list:
                continue

            replace_list[w2] = w1
    
    for key, value in replace_list.items():
        df[what] = df[what].replace(key, value)
    return replace_list