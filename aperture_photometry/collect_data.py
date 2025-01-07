import os
import pandas as pd
import csv
dirpath = os.path.dirname(os.path.abspath(__file__))

imgpath = '/Volumes/PortableSSD/SOFI'
folders = []
for item in os.listdir(imgpath):
    if item[-4:]!='.txt' and item[-5:]!='.fits' and item[-4:]!='.csv':
        folders.append(item)

generalphot = pd.read_csv(f'{imgpath}/generalphotometry.csv')
simpleinfo = pd.read_csv(f'{imgpath}/SimpleInfo.csv')

for folder in folders:
    completedf = pd.read_csv(f'{imgpath}/{folder}/photometry.csv')
    usefuldf = pd.read_csv(f'{imgpath}/{folder}/sn_magnitude.csv')
    generalphot = pd.concat([generalphot, completedf], ignore_index=True)
    simpleinfo = pd.concat([simpleinfo, usefuldf], ignore_index=True)
    

generalphot.to_csv(f'{imgpath}/generalphotometry.csv')
simpleinfo.to_csv(f'{imgpath}/SimpleInfo.csv')