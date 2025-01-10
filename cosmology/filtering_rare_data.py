# importing libraries
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import xlwings as xw
dirpath = os.path.dirname(os.path.abspath(__file__))
data_path = dirpath
# path_redshift = '/Users/j.alcaide/Desktop/Joves i Ciència/article/Hubble computing/without_outliners/copy_redshift.xlsx'
# dist_path = '/Users/j.alcaide/Desktop/Joves i Ciència/article/Hubble computing/without_outliners/copy_I_distances.txt'
jic_2023_path = '/Users/joanalnu/Library/Mobile Documents/com~apple~CloudDocs/Joves i Ciència/article/Hubble computing/without_outliners'
path_redshift = jic_2023_path + "/copy_redshift.xlsx"
dist_path = jic_2023_path + "/copy_I_distances.txt"

def to_rgb(a,b,c):
    return (a/255, b/255, c/255)

# SN
# Read SN data
#df = pd.read_csv(f'{data_path}/COPY_Results_SNCOSMO.csv')
df = pd.read_csv(f'{data_path}/Results_SNCOSMO.csv')
df = df.dropna()

# applying cuts for clear outliners
df = df[df['dm']>0]
df = df[df['dm']<40]
df = df[df['dmerr']>0]
df = df[df['dmerr']<5]

# applying additional cuts
dropping = list()
for index, row in df.iterrows():
    if row[2]>0.1 and row[3]<36:
        dropping.append(index)
    # if row[2]>0.05 and row[2]<0.125:
    #     if row[3] < 36:
    #         dropping.append(index)
df = df.drop(dropping)

# Creating variables
names = np.array(df['name'])
redshifts = np.array(df['redshift'])
dms = np.array(df['dm'])
dm_errs = np.array(df['dmerr'])


# Compute additional magnitudes (velocity & distance)
velocities = redshifts * 299792.458 # km/s
distances = 10**((dms/5)+1) # pc
sigma_distances = distances * np.log(10) / 5 * dm_errs # pc
distances /= 1000000 # to Mpc
sigma_distances /= 1000000 # to Mpc

# remove error in sigma distances (huge, weird errorbars)
new_values = list()
for value in sigma_distances:
    if value > 0.01* (10**6):
        new_values.append(1)
    else:
        new_values.append(value)
sigma_distances = np.array(new_values)


# create csv file with used data (aplying cuts, etc)
df_used = pd.DataFrame({'name':names, 'redshift':redshifts, 'dm':dms, 'dmerr':dm_errs})
df_used.to_csv(f'{dirpath}/used_data.csv', index=False)


df_filtered = pd.DataFrame(columns=['name', 'redshift', 'dm', 'dmerr', 'velocity', 'distance', 'sigma_distance', 'host', 'type', 'coordinates'])
for index, row in df.iterrows():
    if row['redshift'] > 0.06 and row['dm'] < 36:
        df_filtered = df_filtered._append({'name':row['name'], 'redshift':row['redshift'], 'dm':row['dm'], 'dmerr':row['dmerr'], 'velocity':row['redshift'] * 299792.458, 'distance':10**((row['dm']/5)+1)/1000000, 'sigma_distance':10**((row['dm']/5)+1) * np.log(10) / 5 * row['dmerr']/1000000}, ignore_index=True)

from wiserep_api import get_target_property

for index, row in df_filtered.iterrows():
    for prop in ['host', 'type', 'coords']:
        value = get_target_property(row['name'], prop)
        df_filtered.at[index, prop] = value


df_filtered.to_csv(f'{dirpath}/filtered_data.csv', index=False)