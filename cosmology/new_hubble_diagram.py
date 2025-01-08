# importing libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


# define paths
dirpath = os.path.dirname(os.path.abspath(__file__))
data_path = dirpath + "/Results_SNCOSMO.csv"

# read data
df = pd.read_csv(data_path)
df = df.dropna()

# applying cuts
df = df[df['redshift']<0.12]
df = df[df['dm']>32]
df = df[df['dm']<38.5]
# df = df[df['dmerr']>0]
# df = df[df['dmerr']<5]

# additional cuts
# dropping = list()
# for index, row in df.iterrows():
#     if row[2]>0.1 and row[3]<36:
#         dropping.append(index)
#     # if row[2]>0.05 and row[2]<0.125:
#     #     if row[3] < 36:
#     #         dropping.append(index)
# df = df.drop(dropping)

# creating lists
names = np.array(df['name'])
redshifts = np.array(df['redshift'])
dms = np.array(df['dm'])
dm_errs = np.array(df['dmerr'])

# compute further magnitudes
velocities = redshifts * 299792.458 # km/s
distances = 10**((dms/5)+1) # pc
sigma_distances = distances * np.log(10) / 5 * dm_errs # in pc
distances_mpc = distances/1000000 # to Mpc
sigma_distances_mpc = sigma_distances/1000000 # to Mpc

# remove weird errorbars
new_values = list()
for value in sigma_distances:
    if value > 0.01:
        new_values.append(1)
    else:
        new_values.append(value/1000000)
sigma_distances_mpc = np.array(new_values)



#################
# Computing linear fits
def model(x, a, b):
    return a * x + b

# numpy polyfit
npslope, npintercept = np.polyfit(dms, redshifts, 1)
best_polyfit = npslope * dms + npintercept

# numpy with 2nd gen magnitudes
npslope2g, npintercept2g = np.polyfit(distances, velocities, 1)
best_polyfit2g = npslope2g * distances + npintercept2g

# reverse numpy polyfit
rev_npslope, rev_npintercept = np.polyfit(redshifts, dms, 1)
rev_best_polyfit = rev_npslope * redshifts + rev_npintercept

# reverse numpy with 2nd gen magnitudes
rev_npslope2g, rev_npintercept2g = np.polyfit(velocities, distances, 1)
rev_best_polyfit2g = rev_npslope2g * velocities + rev_npintercept2g

# display plots
linear_fig, axs = plt.subplots(2, 2)
axs[0,0].scatter(dms, redshifts)
axs[0,0].plot(dms, best_polyfit, label=f"polyfit $H_0 = {npslope}$ or {(1/npslope)}")
axs[0,0].set_xlabel("$\mu$")
axs[0,0].set_ylabel("$z$")
axs[0,0].legend()

axs[1,0].scatter(distances, velocities)
axs[1,0].plot(distances, best_polyfit2g, label=f"2g polyfit $H_0 = {npslope2g}$ or ({(1/npslope2g)}")
axs[1,0].set_xlabel("$\mu$")
axs[1,0].set_ylabel("$z$")
axs[1,0].legend()

axs[0,1].scatter(redshifts, dms)
axs[0,1].plot(redshifts, rev_best_polyfit, label=f"polyfit $H_0 = {rev_npslope}$ or {(1/rev_npslope)}")
axs[0,1].set_xlabel("$\mu$")
axs[0,1].set_ylabel("$z$")
axs[0,1].legend()

axs[1,1].scatter(velocities, distances)
axs[1,1].plot(velocities, rev_best_polyfit2g, label=f"2g polyfit $H_0 = {rev_npslope2g}$ or ({(1/rev_npslope2g)}")
axs[1,1].set_xlabel("$\mu$")
axs[1,1].set_ylabel("$z$")
axs[1,1].legend()

linear_fig.savefig("hubble_diagram.png")

plt.show()