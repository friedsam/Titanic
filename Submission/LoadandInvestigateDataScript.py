#!/usr/bin/env python
# encoding: utf-8
"""
LoadandInvestigateDataScript.py

Created by Claudia Friedsam on 2014-08-06.
Copyright (c) 2014 __MyCompanyName__. All rights reserved.
"""

import sys
import os
import pandas as pd
import numpy as np
import LoadandInvestigate as li

## load data into a dataframe and display general information
data_file = li.load_data('../train.csv')
## look at data types
## conclusion:
## OK: PassengerID, Survived, PClass, Age, SibSp,Parch, Fare
## Object: Name, Sex, Ticket, Cabin, Embarked
# data_file.dtypes
## confirm:
print data_file.dtypes[data_file.dtypes.map(lambda x: x=='object')]
## determine missing values
## Conclusion:
## missing values: Age, Cabin, Embarked
print data_file.info()
## look at statistical data
## conlusion: not useful at this point
# print data_file.describe()





