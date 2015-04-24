#!/usr/bin/env python
# encoding: utf-8
"""
VisualizeDataScript.py

Created by Claudia Friedsam on 2014-08-07.
Copyright (c) 2014 __MyCompanyName__. All rights reserved.
"""

import sys
import os
import pandas as pd
import numpy as np
import LoadandInvestigate as li
import matplotlib.pyplot as plt
import pandas.tools.rplot as rplot


##load modified data file
df = li.load_data('../train_mod_num.csv')

##get headers as list
vars=df.columns.values.tolist()
#or: vars = list(df)

# #plot histograms
# for var in vars:
# 	plt.figure()
# 	plt.title(var)
# 	df[var].hist(bins=50)
# 	savepath = '/Users/claudia/Documents/Courses/Coursera/DataScience2/KoogleComp/Titanic/Graphs/Histos/' + var + '.png'
# 	plt.savefig(savepath)

# #plot histograms for Survivors
# dfS=df[df['Survived'] == 1]
# for var in vars:
# 	plt.figure()
# 	plt.title(var)
# 	dfS[var].hist(bins=50)
# 	savepath = '/Users/claudia/Documents/Courses/Coursera/DataScience2/KoogleComp/Titanic/Graphs/HistosSurv/' + var + '.png'
# 	plt.savefig(savepath)

# #plot boxplot Fare and Age
# #AgeFill
# bp = df.boxplot(column=['AgeFill'], by='Survived')
# savepath = '/Users/claudia/Documents/Courses/Coursera/DataScience2/KoogleComp/Titanic/Graphs/BoxPlotAge/Survived.png'
# plt.savefig(savepath)
# bp = df.boxplot(column=['AgeFill'], by='Pclass')
# savepath = '/Users/claudia/Documents/Courses/Coursera/DataScience2/KoogleComp/Titanic/Graphs/BoxPlotAge/PClass.png'
# plt.savefig(savepath)
# bp = df.boxplot(column=['AgeFill'], by='SibSp')
# savepath = '/Users/claudia/Documents/Courses/Coursera/DataScience2/KoogleComp/Titanic/Graphs/BoxPlotAge/SibSp.png'
# plt.savefig(savepath)
# bp = df.boxplot(column=['AgeFill'], by='Parch')
# savepath = '/Users/claudia/Documents/Courses/Coursera/DataScience2/KoogleComp/Titanic/Graphs/BoxPlotAge/Parch.png'
# plt.savefig(savepath)
# bp = df.boxplot(column=['AgeFill'], by='Gender')
# savepath = '/Users/claudia/Documents/Courses/Coursera/DataScience2/KoogleComp/Titanic/Graphs/BoxPlotAge/Gender.png'
# plt.savefig(savepath)
# bp = df.boxplot(column=['AgeFill'], by='Cabin_t')
# savepath = '/Users/claudia/Documents/Courses/Coursera/DataScience2/KoogleComp/Titanic/Graphs/BoxPlotAge/Cabin_t.png'
# plt.savefig(savepath)
# bp = df.boxplot(column=['AgeFill'], by='Embarked_t')
# savepath = '/Users/claudia/Documents/Courses/Coursera/DataScience2/KoogleComp/Titanic/Graphs/BoxPlotAge/Embarked_t.png'
# plt.savefig(savepath)
# #Fare
# bp = df.boxplot(column=['Fare'], by='Survived')
# savepath = '/Users/claudia/Documents/Courses/Coursera/DataScience2/KoogleComp/Titanic/Graphs/BoxPlotFare/Survived.png'
# plt.savefig(savepath)
# bp = df.boxplot(column=['Fare'], by='Pclass')
# savepath = '/Users/claudia/Documents/Courses/Coursera/DataScience2/KoogleComp/Titanic/Graphs/BoxPlotFare/PClass.png'
# plt.savefig(savepath)
# bp = df.boxplot(column=['Fare'], by='SibSp')
# savepath = '/Users/claudia/Documents/Courses/Coursera/DataScience2/KoogleComp/Titanic/Graphs/BoxPlotFare/SibSp.png'
# plt.savefig(savepath)
# bp = df.boxplot(column=['Fare'], by='Parch')
# savepath = '/Users/claudia/Documents/Courses/Coursera/DataScience2/KoogleComp/Titanic/Graphs/BoxPlotFare/Parch.png'
# plt.savefig(savepath)
# bp = df.boxplot(column=['Fare'], by='Gender')
# savepath = '/Users/claudia/Documents/Courses/Coursera/DataScience2/KoogleComp/Titanic/Graphs/BoxPlotFare/Gender.png'
# plt.savefig(savepath)
# bp = df.boxplot(column=['Fare'], by='Cabin_t')
# savepath = '/Users/claudia/Documents/Courses/Coursera/DataScience2/KoogleComp/Titanic/Graphs/BoxPlotFare/Cabin_t.png'
# plt.savefig(savepath)
# bp = df.boxplot(column=['Fare'], by='Embarked_t')
# savepath = '/Users/claudia/Documents/Courses/Coursera/DataScience2/KoogleComp/Titanic/Graphs/BoxPlotFare/Embarked_t.png'
# plt.savefig(savepath)


# #plot bar charts
# savepath = '/Users/claudia/Documents/Courses/Coursera/DataScience2/KoogleComp/Titanic/Graphs/BarCharts/'
# 
# bc = df.groupby(['Gender', 'Pclass']).Survived.sum().plot(kind='barh')
# plt.savefig(savepath + 'GenderPclassbarh.png')
# 
# bc = df.groupby(['Gender', 'Pclass', 'Cabin_t']).Survived.sum().plot(kind='barh')
# plt.savefig(savepath + 'GenderPclassCabinbarh.png')
# 
# bc = df.groupby(['Gender', 'Pclass', 'Embarked_t']).Survived.sum().plot(kind='barh')
# plt.savefig(savepath + 'GenderPclassEmbarkedbarh.png')
# 
# death_counts = pd.crosstab([df.Pclass, df.Gender], df.Survived.astype(bool))
# death_counts.plot(kind='bar', stacked=True, color=['blue','red'], grid=False)
# plt.savefig(savepath + 'GenderPclasscrosstab.png')
# death_counts.div(death_counts.sum(1).astype(float), axis=0).plot(kind='barh', stacked= True, color=['blue','red'])
# plt.savefig(savepath + 'GenderPclasscrosstabnorm.png')
# 
# death_counts = pd.crosstab([df.Pclass, df.Gender, df.Cabin_t], df.Survived.astype(bool))
# death_counts.plot(kind='bar', stacked=True, color=['blue','red'], grid=False)
# plt.savefig(savepath + 'GenderPclassCabincrosstab.png')
# death_counts.div(death_counts.sum(1).astype(float), axis=0).plot(kind='barh', stacked= True, color=['blue','red'])
# plt.savefig(savepath + 'GenderPclassCabincrosstabnorm.png')
# 
# death_counts = pd.crosstab([df.Pclass, df.Gender, df.Embarked_t], df.Survived.astype(bool))
# death_counts.plot(kind='bar', stacked=True, color=['blue','red'], grid=False)
# plt.savefig(savepath + 'GenderPclassEmbarkedcrosstab.png')
# death_counts.div(death_counts.sum(1).astype(float), axis=0).plot(kind='barh', stacked= True, color=['blue','red'])
# plt.savefig(savepath + 'GenderPclassEmbarkedcrosstabnorm.png')



#plot trellis plots
savepath = '/Users/claudia/Documents/Courses/Coursera/DataScience2/KoogleComp/Titanic/Graphs/TrellisPlots/'
df_mod = df[df.Embarked_t>0]
df_S = df[(df.Survived == 1) & (df.Embarked_t > 0)]

#Age
plt.figure()
tp = rplot.RPlot(df, x='AgeFill')
tp.add(rplot.TrellisGrid(['Gender', 'Pclass']))
tp.add(rplot.GeomHistogram())
tp.render(plt.gcf())
plt.savefig(savepath +'AgeGPclAll.png')

plt.figure()
tp = rplot.RPlot(df_S, x='AgeFill')
tp.add(rplot.TrellisGrid(['Gender', 'Pclass']))
tp.add(rplot.GeomHistogram())
tp.render(plt.gcf())
plt.savefig(savepath +'AgeGPclS.png')

plt.figure()
tp = rplot.RPlot(df, x='AgeFill')
tp.add(rplot.TrellisGrid(['Gender', 'Cabin_t']))
tp.add(rplot.GeomHistogram())
tp.render(plt.gcf())
plt.savefig(savepath +'AgeGCabAll.png')

plt.figure()
tp = rplot.RPlot(df_S, x='AgeFill')
tp.add(rplot.TrellisGrid(['Gender', 'Cabin_t']))
tp.add(rplot.GeomHistogram())
tp.render(plt.gcf())
plt.savefig(savepath +'AgeGCabS.png')

plt.figure()
tp = rplot.RPlot(df_mod, x='AgeFill')
tp.add(rplot.TrellisGrid(['Gender', 'Embarked_t']))
tp.add(rplot.GeomHistogram())
tp.render(plt.gcf())
plt.savefig(savepath +'AgeGEmAll.png')

plt.figure()
tp = rplot.RPlot(df_S, x='AgeFill')
tp.add(rplot.TrellisGrid(['Gender', 'Embarked_t']))
tp.add(rplot.GeomHistogram())
tp.render(plt.gcf())
plt.savefig(savepath +'AgeGEmS.png')

#Fare
plt.figure()
tp = rplot.RPlot(df, x='Fare')
tp.add(rplot.TrellisGrid(['Gender', 'Pclass']))
tp.add(rplot.GeomHistogram())
tp.render(plt.gcf())
plt.savefig(savepath +'FareGPclAll.png')

plt.figure()
tp = rplot.RPlot(df_S, x='Fare')
tp.add(rplot.TrellisGrid(['Gender', 'Pclass']))
tp.add(rplot.GeomHistogram())
tp.render(plt.gcf())
plt.savefig(savepath +'FareGPclS.png')

plt.figure()
tp = rplot.RPlot(df, x='Fare')
tp.add(rplot.TrellisGrid(['Gender', 'Cabin_t']))
tp.add(rplot.GeomHistogram())
tp.render(plt.gcf())
plt.savefig(savepath +'FareGCabAll.png')

plt.figure()
tp = rplot.RPlot(df_S, x='Fare')
tp.add(rplot.TrellisGrid(['Gender', 'Cabin_t']))
tp.add(rplot.GeomHistogram())
tp.render(plt.gcf())
plt.savefig(savepath +'FareGCabS.png')

plt.figure()
tp = rplot.RPlot(df_mod, x='Fare')
tp.add(rplot.TrellisGrid(['Gender', 'Embarked_t']))
tp.add(rplot.GeomHistogram())
tp.render(plt.gcf())
plt.savefig(savepath +'FareGEmAll.png')

plt.figure()
tp = rplot.RPlot(df_S, x='Fare')
tp.add(rplot.TrellisGrid(['Gender', 'Embarked_t']))
tp.add(rplot.GeomHistogram())
tp.render(plt.gcf())
plt.savefig(savepath +'FareGEmS.png')



















