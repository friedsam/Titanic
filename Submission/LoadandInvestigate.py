#!/usr/bin/env python
# encoding: utf-8
"""
LoadandInvestigateData.py

Created by Claudia Friedsam on 2014-08-06.
Copyright (c) 2014 __MyCompanyName__. All rights reserved.
"""

import sys
import os
import pandas as pd
import numpy as np

def load_data(file):
	#read csv_file
	data_file = pd.read_csv(file, header=0)
	return data_file






