# make the final list
import pandas as pd
import os
dirpath = os.path.dirname(os.path.abspath(__file__))


df = pd.read_csv(os.path.join(dirpath, 'SNIa-91bg-like_cleaned.csv'))
Type = 'SNIa-91bg-like'

final_wn = list()
final_do = list()
final_co = list()
final_fn = list()
final_type = list()

folder = '/Volumes/PortableSSD/SNIa-91bg-like_processed/'
contents = os.listdir(folder)
files = [content[:-5] for content in contents if content[:3] == 'ADP']

for file in files:
    rows = df[df['ARCFILE'] == file]
    print(rows)
    
    for index, row in rows.iterrows():
        wn = row['Wiserep_Name']
        do = row['Date Obs_2']
        co = row['Uploaded file input value']
        fn = row['ARCFILE']
        
        print(f'{wn}\t{do}\t{co}\t{fn}')
        
        final_wn.append(wn)
        final_do.append(do)
        final_co.append(co)
        final_fn.append(fn)
        final_type.append(Type)

final_file_path= os.path.join(dirpath, 'SNIa-91bg-like_ProcessedList.txt')
with open(final_file_path, 'w') as final_file:
    final_file.write(f'WiserepName\tDateObs\tCoordinates\tFitsName\tSNType')
    for i in range(len(final_wn)):
        final_file.write(f'{final_wn[i]}\t{final_do[i]}\t{final_co[i]}\t{final_fn[i]}\t{final_type[i]}\n')

















# # for each fits file in portableSSD make a copy of the file with the name: 'SNNAME_ARCFILE.fits'
# import pandas as pd
# from astropy.io import fits
# import os
# dirpath = os.path.dirname(os.path.abspath(__file__))


# df = pd.read_csv(dirpath+'/SNIa_cleaned.csv')
# arcfile_column = df['ARCFILE']
# object_column = df['Wiserep_Name']

# imagepath = '/Volumes/PortableSSD'
# folder = '/SNIa-91T-like_processed/'
# folder_files = os.listdir(imagepath+folder)

# saving_folder = '/Names_SNIa_processed/' #change for savings
# same = False

# fits_files = list()
# for file in folder_files:
#     if file[0] =='A':
#         fits_files.append(file)

# not_found = list()
# for file in fits_files:
#     for i in range(len(arcfile_column)):
#         if arcfile_column[i] == file.rstrip('.fits'):
#             same = True
#             hdul = fits.open(f'{imagepath}{folder}{file}')
#             hdul.writeto(f'{imagepath}{saving_folder}{object_column[i]}_{file}', overwrite=True)
#             print(f'Same file saved in {imagepath}{folder}{object_column[i]}{file}')
#     if same == False:
#         not_found.append(file)
#     else:
#         same = False

# print(not_found)
# print(f'{len(arcfile_column)} \t {len(fits_files)} \t {len(object_column)}')
# print(f'{len(fits_files)} \t \t {len(os.listdir(imagepath+saving_folder))}')

# # for file in fits_files:
# #     content = os.listdir(imagepath+saving_folder)
# #     if file in content:
# #         print(f'{file} in content')
# #     else:
# #         print(f'{file} is missing')