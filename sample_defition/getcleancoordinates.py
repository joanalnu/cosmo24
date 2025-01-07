# for SNIa-...-like
# import os
# import pandas as pd
# dirpath = os.path.dirname(os.path.abspath(__file__))

# df = pd.read_csv(dirpath+'/SNIa-91T-like_cleaned.csv')
# names = df['Wiserep_Name']

# coords_file = open(dirpath+'/coords1/SNIa-91T-likecoordsnames.txt', 'r')
# coords = list()
# for name in names:
#     coords_file.seek(0)
#     for line in coords_file:
#         line = line.split()
#         #print(f'{name}     {line}')
#         if name == line[2].lstrip('#'):
#             coords.append(f'{line[0]} {line[1]}')
#             print(f'{name}     {line}')

# file = open(dirpath+'/SNIa-91T-like_cleancoords.txt', 'w')
# for coord in coords:
#     file.write(f'{coord}\n')

# for SNIa
import os
import pandas as pd
dirpath = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(dirpath+'/SNIa_cleaned.csv')
column = df['Uploaded file input value']

coords = list()
for value in column:
    if value[0]!='#':
        coords.append(value)

file = open(dirpath+'/SNIa_cleancoords.txt', 'w')
for coord in coords:
    file.write(f'{coord}\n')
