import os
import pandas as pd
dirpath = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(dirpath+'/SNIa_cleaned.csv')
names = df['Wiserep_Name']
droping_indexs = list()
count = 0
for i in range(len(names)):
    name = names[i]
    if type(name) == str:
        if name[:4] == "SNLS":
            droping_indexs.append(i)
            print(f'deleted {i}  \t \t \t \t \t\t  {count} lines out of {i} were deleted')
            count +=1

df = df.drop(droping_indexs)

df.to_csv(dirpath+'/SNIa_noSNLS_cleaned.csv', index=False)

print(f'{count} lines out of {len(names)} were deleted')




# # cleverer code from internet
# import os
# import pandas as pd

# # Define the directory path and read the CSV file
# dirpath = os.path.dirname(os.path.abspath(__file__))
# df = pd.read_csv(dirpath + '/SNIa_cleaned.csv')

# # Collect the indices of rows to drop
# drop_indices = [i for i, name in enumerate(df['Wiserep_Name']) if name[:4] == "SNLS"]

# # Drop the collected indices
# df = df.drop(drop_indices)

# # Save the cleaned DataFrame to a new CSV file
# df.to_csv(dirpath + '/SNIa_noSNLS_cleaned.csv', index=False)

# # Print the number of lines deleted
# print(f'{len(drop_indices)} lines out of {len(df) + len(drop_indices)} were deleted')
