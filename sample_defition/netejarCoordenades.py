# line example
# Target[15:05:30.072	+01:38:02.40]: A total of 25 records were found matching the provided criteria(15:05:30.072 +01:38:02.40).
import os
folderpath = os.path.dirname(os.path.abspath(__file__))

coords3 = list()
coords4 = list()
with open('/Users/j.alcaide/Documents/ICE_FASE2JiC/SNIacoords_net_eso2.txt', 'r') as file:
    for line in file:
        line = file.readline()
        if line[0]=='#' and len(line) > 2 and line[2]=='T':
            line_sep = line.split()
            if line_sep[6] != '0':
                first_part = line_sep[1].lstrip('Target[')
                second_part = line_sep[2].rstrip(']:')
                coords3.append(f'{first_part}\t{second_part}\n')
            elif line_sep[6]=='0':
                first_part = line_sep[1].lstrip('Target[')
                second_part = line_sep[2].rstrip(']:')
                coords4.append(f'{first_part}\t{second_part}\n')

with open(folderpath+'/coords3/SNIacoords_net.txt', 'w') as file:
    for coord in coords3:
        file.write(f'{coord}')

with open(folderpath+'/coords4/SNIacoords_net_txt', 'w') as file:
    for coord in coords4:
        file.write(f'{coord}')
    




# # coords1 -> coords2
# import os
# folderpath = os.path.dirname(os.path.abspath(__file__))

# coords2 = list()
# with open('/Users/j.alcaide/Documents/ICE_FASE2JiC/SNIa-91bg-likecoords_eso.txt', 'r') as file:
#     for line in file:
#         line = file.readline()
#         if line[0]=='#' and len(line) > 2 and line[2]=='T':
#             line_sep = line.split()
#             if line_sep[6] != '0':
#                 first_part = line_sep[1].lstrip('Target[')
#                 second_part = line_sep[2].rstrip(']:')
#                 coords2.append(f'{first_part}\t{second_part}\n')

# with open(folderpath+'/coords2/SNIa-91bg-likecoords_net.txt', 'w') as file:
#     for coord in coords2:
#         file.write(f'{coord}')



# coords2 -> coords3 & coords4
# import os
# folderpath = os.path.dirname(os.path.abspath(__file__))

# coords3 = list()
# coords4 = list()
# with open('/Users/j.alcaide/Documents/ICE_FASE2JiC/SNIacoords_net_eso2.txt', 'r') as file:
#     for line in file:
#         line = file.readline()
#         if line[0]=='#' and len(line) > 2 and line[2]=='T':
#             line_sep = line.split()
#             if line_sep[6] != '0':
#                 first_part = line_sep[1].lstrip('Target[')
#                 second_part = line_sep[2].rstrip(']:')
#                 coords3.append(f'{first_part}\t{second_part}\n')
#             elif line_sep[6]=='0':
#                 first_part = line_sep[1].lstrip('Target[')
#                 second_part = line_sep[2].rstrip(']:')
#                 coords4.append(f'{first_part}\t{second_part}\n')

# with open(folderpath+'/coords3/SNIacoords_net.txt', 'w') as file:
#     for coord in coords3:
#         file.write(f'{coord}')

# with open(folderpath+'/coords4/SNIacoords_net_txt', 'w') as file:
#     for coord in coords4:
#         file.write(f'{coord}')




#coords2 -> coords3 & coords4 for SNIa list
# import os
# folderpath = os.path.dirname(os.path.abspath(__file__))

# coords3 = list()
# coords4 = list()
# with open('/Users/j.alcaide/Documents/ICE_FASE2JiC/SNIacoords_net_eso2.txt', 'r') as file:
#     for line in file:
#         line_sep = line.split()
#         if line[0]=='#' and len(line) > 2:
#             if line[2]=='T' and line_sep[6]!='0':
#                 first_part = line_sep[1].lstrip('Target[')
#                 second_part = line_sep[2].rstrip(']:')
#                 coords3.append(f'{first_part}\t{second_part}\n')
#             elif line[2]=='T' and line_sep[6]=='0':
#                 first_part = line_sep[1].lstrip('Target[')
#                 second_part = line_sep[2].rstrip(']:')
#                 coords4.append(f'{first_part}\t{second_part}\n')

# with open(folderpath+'/coords3/SNIacoords_net.txt', 'w') as file:
#     for coord in coords3:
#         file.write(f'{coord}')

# with open(folderpath+'/coords4/SNIacoords_net.txt', 'w') as file:
#     for coord in coords4:
#         file.write(f'{coord}')
#         print(coord)