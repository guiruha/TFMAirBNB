#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 20:20:21 2020

@author: guillem
"""


import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv('~/DadesAirBNB/Metro/TRANSPORTS.csv')

df.columns

df = df[['NOM_CAPA', 'LONGITUD', 'LATITUD', 'NOM_BARRI', 'EQUIPAMENT']]

df['NOM_CAPA'].value_counts()

df = df[(df['NOM_CAPA']!='Funicular')&(df['NOM_CAPA'] != 'Telefèric')]

df['NOM_BARRI'].value_counts()

fig, ax = plt.subplots(1, 1, figsize = (20, 15))
sns.scatterplot('LONGITUD', 'LATITUD', hue = 'NOM_CAPA', data = df, ax = ax)
plt.show()
