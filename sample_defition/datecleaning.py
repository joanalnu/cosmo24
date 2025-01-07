# #main date cleaning for SNIa_processed
import os
import pandas as pd
dirpath = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(dirpath+'/SNIa_noSNLS_cleaned.csv')

orig_wn = df['Wiserep_Name']
orig_do = df['Date Obs_2']

normal_names_count = 0
filtered_wn = list()
for i in range(len(orig_wn)):
    name = orig_wn[i]
    if type(name) == str and (name[0]=='1' or name[0]=='2'):
        filtered_wn.append(name[:4])
        normal_names_count += 1
    else: 
        #droping_indexs.append(i)
        filtered_wn.append('1000') # PORSIACASOappending a random number to name which aren't SNXXXXabc => not considering those names

filtered_do = list()
for do in orig_do:
    if type(do) != float:
        if (do[0]=='1' or do[0]=='2'):
            filtered_do.append(do[:4])
        else: filtered_wn.append(' ')

print(f'filtered_do: {filtered_do} \n filtered_wn: {filtered_wn}')

trues = 0
mid_trues=0
for i in range(len(filtered_do)):
    if (filtered_do==' ' or filtered_wn==' ') or (filtered_do==' ' and filtered_wn==' '):
        continue # droping_indexs.append(i) # esto lo arregle Mon Jun 17, para que quite las que no tienen nombre SN....
    # elif (type(filtered_do)!=str or type(filtered_wn!=str)) or (type(filtered_do)!=str and type(filtered_wn!=str)):
        # print()
    else:
        if filtered_do[i] == filtered_wn[i]:
            trues+=1
            print(f'normal one: {filtered_do[i]} with {filtered_do[i]}')

        if filtered_wn[i][:3] + (str(int(filtered_wn[i][-1:])+1)) == filtered_do[i]:
            mid_trues +=1
            print(f'big one: {filtered_wn[i][:3] + (str(int(filtered_wn[i][-1:])+1))} with {filtered_do[i]}')

droping_indexs = list()
for i in range(len(filtered_do)):
    if filtered_do[i] != filtered_wn[i] and filtered_do[i][:3] + (str(int(filtered_do[i][-1:])+1)) != filtered_wn[i] and filtered_do[i][:3] + (str(int(filtered_do[i][-1:])-1)) != filtered_wn[i]:
        droping_indexs.append(i)

df = df.drop(droping_indexs)

df.to_csv(dirpath+'/SNIa_cleaned.csv', index=False)
        

print(f'SNe nº: {normal_names_count}')
print(f'anormal nº: {len(filtered_do)-normal_names_count}')
print(f'Trues: {trues} \t \t; falses: {len(filtered_do)-trues}')
print(f'mid-trues: {mid_trues} \t \t; flases: {len(filtered_do)-mid_trues}')
print(f'total SNe: {len(filtered_do)}')





# add wiserep names to csv files (91T and 91bg edition)
# import os
# import pandas as pd
# dirpath = os.path.dirname(os.path.abspath(__file__))

# df = pd.read_csv(dirpath+'/SNIa-91T-like_processed.csv')
# first_column = df['Uploaded file input value']
# csv_coords = list()
# for value in first_column:
#     if type(value) == str and value[0] != '#':
#         values = value.split()
#         print(values)
#         value = values[1]
#         if (value[0]=='+' or value[0]=='-'):
#             csv_coords.append(value)

# new_values = list()
# txt_file = open(dirpath+'/coords1/SNIa-91T-likecoordsnames.txt')
# for coord in csv_coords:
#     txt_file.seek(0)
#     for line in txt_file:
#         line = line.split()
#         if coord == line[1]:
#             new_values.append(line[2].lstrip('#'))


# print(len(new_values), len(first_column), len(csv_coords))

# for i in range(len(first_column)-len(new_values)):
#     new_values.append('')

# df['Wiserep_Name'] = new_values

# df.to_csv(dirpath+'/SNIa-91T-like_processed.csv')

# # add wiserep names to csv files (SNIa edition)
# import os
# import pandas as pd
# import csv
# dirpath = os.path.dirname(os.path.abspath(__file__))

# df = pd.read_csv("SNIa_cleaned.csv")
# first_column = df.iloc[:, 0]

# csv_coords = list()
# for value in first_column:
#     if type(value)==str and value[0]!='#':
#         csv_coords.append(str(value))

# sep_coords = list()
# for coord in csv_coords:
#     sep_coords.append(coord.split(' '))

# print(f'First_column: {first_column}\n')
# print(f'csv_coords: {csv_coords}\n')
# print(f'Sep_coords: {sep_coords}\n')

# new_values = list()
# txt_file = open(dirpath+'/coords1/SNIacoordsnames.txt')
# for coord in sep_coords:
#     txt_file.seek(0)
#     for line in txt_file:
#         line = line.split()
#         if coord[1] == line[1] and coord[0] == line[0]:
#             new_values.append(line[2].lstrip('#'))


# for i in range(len(first_column)-len(new_values)):
#     new_values.append('')

# print(f'New_values: {new_values}')
# print(len(first_column), len(csv_coords), len(sep_coords), len(new_values))

# df['Wiserep Name'] = new_values

# df.to_csv(dirpath+'/SNIa_cleaned.csv')


# # # add wiserep names to csv files (raw version)
# import os
# import pandas as pd
# import csv
# dirpath = os.path.dirname(os.path.abspath(__file__))

# df = pd.read_csv("Raw_SNIacoords_net_eso.csv")
# first_column = df.iloc[:, 0]

# csv_coords = list()
# for value in first_column:
#     if type(value)==str and value[0]!='#':
#         csv_coords.append(str(value))

# sep_coords = list()
# for coord in csv_coords:
#     sep_coords.append(coord.split(' '))

# print(f'First_column: {first_column}\n')
# print(f'csv_coords: {csv_coords}\n')
# print(f'Sep_coords: {sep_coords}\n')

# new_values = list()
# txt_file = open(dirpath+'/coords1/SNIacoordsnames.txt')
# for coord in sep_coords:
#     txt_file.seek(0)
#     for line in txt_file:
#         line = line.split()
#         if coord[1] == line[1] and coord[0] == line[0]:
#             new_values.append(line[2].lstrip('#'))


# for i in range(len(first_column)-len(new_values)):
#     new_values.append('')

# print(f'New_values: {new_values}')
# print(len(first_column), len(csv_coords), len(sep_coords), len(new_values))

# df['Wiserep Name'] = new_values

# df.to_csv(dirpath+'/Raw_SNIa_cleaned.csv')










# # main datecleaning for raw
# import os
# import pandas as pd
# dirpath = os.path.dirname(os.path.abspath(__file__))

# df = pd.read_csv(dirpath+'/Raw_SNIa_cleaned.csv')
# orig_wn = df['Wiserep Name']
# orig_do = df['Release_Date']

# normal_names_count = 0
# filtered_wn = list()
# for name in orig_wn:
#     if type(name) == str and (name[0]=='1' or name[0]=='2'):
#         filtered_wn.append(name[:4])
#         normal_names_count += 1
#     else: filtered_wn.append(' ')

# filtered_do = list()
# for do in orig_do:
#     if type(do) != float:
#         if (do[0]=='1' or do[0]=='2'):
#             filtered_do.append(do[:4])
#         else: filtered_wn.append(' ')

# trues = 0
# mid_trues=0
# for i in range(len(filtered_do)):
#     if filtered_do[i] == filtered_wn[i]:
#         trues+=1
#         print(f'normal one: {filtered_do[i]}')

#     if filtered_do[i][:3] + (str(int(filtered_do[i][-1:])+1)) == filtered_wn[i]:
#         mid_trues +=1
#         print(f'big one: {filtered_do[i][:3] + (str(int(filtered_do[i][:-1])+1))}')

# droping_indexs = list()
# for i in range(len(filtered_do)):
#     if filtered_do[i] != filtered_wn[i] and filtered_do[i][:3] + (str(int(filtered_do[i][-1:])+1)) != filtered_wn[i] and filtered_do[i][:3] + (str(int(filtered_do[i][-1:])-1)) != filtered_wn[i]:
#         droping_indexs.append(i)

# df = df.drop(droping_indexs)

# df.to_csv(dirpath+'/Raw_SNIa_onlytodate.csv', index=False)
        

# print(f'SNe nº: {normal_names_count}')
# print(f'anormal nº: {len(filtered_do)-normal_names_count}')
# print(f'Trues: {trues} \t \t; falses: {len(filtered_do)-trues}')
# print(f'mid-trues: {mid_trues} \t \t; falses: {len(filtered_do)-mid_trues}')
# print(f'total SNe: {len(filtered_do)}')