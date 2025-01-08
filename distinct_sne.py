import os

cloud_path = "/users/joanalnu/Library/Mobile Documents/com~apple~CloudDocs/ICE_FASE2JiC/"

# read item lists from finder
sncosmo_path = cloud_path + "SNCOSMO/"
sncosmo_lst = os.listdir(sncosmo_path)

snoopy_path = cloud_path + "SNPY/"
snoopy_lst = os.listdir(snoopy_path)


# filter out non forsncosmo.txt files
pass_sncosmo = []
pass_snoopy = []
for file in sncosmo_lst:
    if "forsncosmo.txt" in file:
        print(file[6:][:-15])
        pass_sncosmo.append(file[6:][:-15])

for file in snoopy_lst:
    if ".snpy" in file:
        print(file[:-5])
        pass_snoopy.append(file[:-5])

# count distinct items
joint_lst = pass_sncosmo + pass_snoopy
number_of_distinct_sne = 0
repeated = list()
already = list()
for name in joint_lst:
    if name in already:
        number_of_distinct_sne += 1
        repeated.append(name)
    else:
        already.append(name)

if len(already) != (len(pass_sncosmo) + len(pass_snoopy))-number_of_distinct_sne:
    print(f'ValueError: already != calc')

print(f'There were {number_of_distinct_sne} repeated names.')
print(f'This leads to a total of {(len(pass_sncosmo) + len(pass_snoopy))-number_of_distinct_sne} distinct SNe')
print(f'The repeated items were: {repeated}')