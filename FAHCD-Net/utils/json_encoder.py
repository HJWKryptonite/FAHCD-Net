#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/22 0:25
# @Author  : HuJiwei
# @File    : json_encoder.py
# @Software: PyCharm
# @Project: AlignDiff
import datetime
import decimal
import json

import numpy as np


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, decimal.Decimal):
            return float(obj)
        elif isinstance(obj, datetime.datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, datetime.date):
            return obj.strftime("%Y-%m-%d")
        else:
            return str(obj)

