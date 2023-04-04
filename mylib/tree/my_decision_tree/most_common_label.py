import numpy as np


def most_common_label(priori: dict):
    return sorted(list(priori.items()), key=lambda x: x[-1])[-1][0]
