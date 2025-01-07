from wiserep_api.search import download_sn_list
from wiserep_api import get_target_property
import os
folderpath = os.path.dirname(os.path.abspath(__file__))

#download_sn_list("SN Ia")
#download_sn_list("SN Ia-91bg-like")
#download_sn_list("SN Ia-91T-like")

names = list()
with open(folderpath+'/SNIa-91T-like_wiserep.txt', 'r') as file:
    for item in file:
        names.append(item[:-1])

CO = open(folderpath+'/SNIa-91T-likecoords.txt', 'w')
CN = open(folderpath+'/SNIa-91T-likecoordsnames.txt', 'w')
CNone = open(folderpath+'/SNIa-91T-likenonecoordsnames.txt', 'w')
for name in names:
    value = get_target_property(name, 'coords')
    if type(value) != str:
        CNone.write(f'{name}\n')
    else:
        CO.write(f'{value}\n')
        CN.write(f'{value} #{name}\n')


names = list()
with open(folderpath+'/SNIa-91bg-like_wiserep.txt', 'r') as file:
    for item in file:
        names.append(item[:-1])

CO = open(folderpath+'/SNIa-91bg-likecoords.txt', 'w')
CN = open(folderpath+'/SNIa-91bg-likecoordsnames.txt', 'w')
CNone = open(folderpath+'/SNIa-91bg-likenonecoordsnames.txt', 'w')
for name in names:
    value = get_target_property(name, 'coords')
    if type(value) != str:
        CNone.write(f'{name}\n')
    else:
        CO.write(f'{value}\n')
        CN.write(f'{value} #{name}\n')