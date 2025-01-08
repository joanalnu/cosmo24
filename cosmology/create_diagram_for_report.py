# importing libraries
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import xlwings as xw

# defining important paths
dirpath = os.path.dirname(os.path.abspath(__file__))



# Read SN data
df = pd.read_csv(f'{dirpath}/Results_SNCOSMO.csv')
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

# Creating variable arrays
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


# define redshift array
z=np.linspace(0.001, 0.17, 1000)


# Compute fits with COSMOLOGY
from astropy.cosmology import FlatLambdaCDM, LambdaCDM

# define models
cosmo1 = FlatLambdaCDM(H0=70, Om0=0.3)
cosmo2 = LambdaCDM(H0=70, Om0=0.3, Ode0=1.2)
cosmo3 = LambdaCDM(H0=100, Om0=0.3, Ode0=0.7)

# Compute distmods
distmod1 = cosmo1.distmod(z).value
distmod2 = cosmo2.distmod(z).value
distmod3 = cosmo3.distmod(z).value

# plot cosmological models
plt.figure()
plt.scatter(redshifts, dms, s=12)
plt.errorbar(redshifts, dms, yerr=[dm_errs], fmt='none')

# plot cosmo models
plt.plot(z, distmod1, label='Flat $\Lambda$CDM: $H_0 = 70 km s^{-1} Mpc^{-1}$; $\Omega_{m_0} = 0.3$', color=(239/255, 134/255, 54/255))
plt.plot(z, distmod2, label='$\Lambda$CDM: $H_0 = 70 km s^{-1} Mpc^{-1}$; $\Omega_{m_0} = 0.3$; $\Omega_{\Lambda} = 1.2$', color=(187/255,58/255,50/255))
plt.plot(z, distmod3, label='$\Lambda$CDM: $H_0 = 50 km s^{-1} Mpc^{-1}$; $\Omega_{m_0} = 0.3$; $\Omega_{\Lambda} = 0.7$', color=(81/255,197/255,58/255))

# personalize
plt.title('Cosmological models')
plt.legend()
plt.xlabel('redshift $z$')
plt.ylabel('distance modulus $\mu$')

# save
plt.savefig('second_actual_fig.png', dpi=300)