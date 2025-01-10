# importing libraries
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import xlwings as xw
dirpath = os.path.dirname(os.path.abspath(__file__))
data_path = dirpath
# path_redshift = '/Users/j.alcaide/Desktop/Joves i Ciència/article/Hubble computing/without_outliners/copy_redshift.xlsx'
# dist_path = '/Users/j.alcaide/Desktop/Joves i Ciència/article/Hubble computing/without_outliners/copy_I_distances.txt'
jic_2023_path = '/Users/joanalnu/Library/Mobile Documents/com~apple~CloudDocs/Joves i Ciència/article/Hubble computing/without_outliners'
path_redshift = jic_2023_path + "/copy_redshift.xlsx"
dist_path = jic_2023_path + "/copy_I_distances.txt"

def to_rgb(a,b,c):
    return (a/255, b/255, c/255)

# SN
# Read SN data
#df = pd.read_csv(f'{data_path}/COPY_Results_SNCOSMO.csv')
# df = pd.read_csv(f'{data_path}/Results_SNCOSMO.csv')
df = pd.read_csv('/Users/joanalnu/Downloads/trucked.csv')
df = df.dropna()
print(df)

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

# Creating variables
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


# create csv file with used data (aplying cuts, etc)
from wiserep_api import get_target_property
hosts = list()
types = list()
coords = list()
for name in names:
    hosts.append(get_target_property(name, 'host'))
    types.append(get_target_property(name, 'type'))
    coords.append(get_target_property(name, 'coords'))

df_used = pd.DataFrame({'name':names, 'redshift':redshifts, 'dm':dms, 'dmerr':dm_errs, 'host':hosts, 'type':types, 'coordinates':coords})

df_used.to_csv(f'{dirpath}/used_data.csv', index=False)


# Cepheid data (Alcaide 2023)
# Create Cepheid dataframe
cep_df = pd.DataFrame(columns=['name','redshift','redshifterr','V_dist','V_dist_err','I_dist','I_dist_err'])

# read redshifts
# open file
wb = xw.Book(path_redshift)
sheet = wb.sheets[0]
red_names = list()
cep_red = list()
cep_red_err = list()
names_cell = sheet['A1']
red_cell = sheet['B1']
err_cell = sheet['C1']
# iterate through spreadsheet
while names_cell.value:
    red_names.append(names_cell.value)
    cep_red.append(float(red_cell.value))
    cep_red_err.append(float(err_cell.value))
    names_cell = names_cell.offset(1,0)
    red_cell = red_cell.offset(1,0)
    err_cell = err_cell.offset(1,0)
# copy to cep_df
cep_df['name'] = red_names
cep_df['redshift'] = cep_red
cep_df['redshifterr'] = cep_red_err

# reading I distances
with open(dist_path, 'r') as f:
    content = f.readlines()
    for line in content:
        line = line.split()
        if line[0]!='#':
            cep_df.loc[cep_df['name']==line[0].replace('_',' '), 'I_dist'] = float(line[1])/1000000 # Mpc
            cep_df.loc[cep_df['name']==line[0].replace('_',' '), 'I_dist_err'] = float(line[1])/1000000-float(line[2])/1000000 # Mpc

# reading V distances
# dist_path = '/Users/j.alcaide/Desktop/Joves i Ciència/article/Hubble computing/without_outliners/copy_V_distances.txt'
dist_path = "/Users/joanalnu/Library/Mobile Documents/com~apple~CloudDocs/Joves i Ciència/article/Hubble computing/without_outliners/copy_V_distances.txt"
with open(dist_path, 'r') as f:
    content = f.readlines()
    for line in content:
        line = line.split()
        if line[0]!='#':
            cep_df.loc[cep_df['name']==line[0].replace('_',' '), 'V_dist'] = float(line[1])/1000000 # Mpc
            cep_df.loc[cep_df['name']==line[0].replace('_',' '), 'V_dist_err'] = float(line[1])/1000000-float(line[2])/1000000 # Mpc

cep_df = cep_df.dropna()
# asign variables
redshifts2 = np.array(cep_df['redshift'])
red_err2 = np.array(cep_df['redshifterr'])
v_dist = np.array(cep_df['V_dist'])
v_dist_err = np.array(cep_df['V_dist_err'])
i_dist = np.array(cep_df['I_dist'])
i_dist_err = np.array(cep_df['I_dist_err'])

# compute other magnitudes (velocities)
vel2 = redshifts2 * 299792.458 # km/s
# vel2_err = red_err2


# Plotting
# redshifts, dms = (list(t) for t in zip(*sorted(zip(redshifts, dms))))
print(redshifts)
print(dms)
print(len(redshifts))
print(len(dms))
redshifts = np.array(redshifts)
dms = np.array(dms)
plt.scatter(redshifts, dms, 'o', color=to_rgb(0, 0, 0), label='SN')
plt.show()


#### new fitting model
from scipy.integrate import quad
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 299792.458  # Speed of light in km/s

# Luminosity distance function for a single redshift value
def luminosity_distance_single(z, H0, Omega_m, Omega_L):
    def integrand(z_prime, Omega_m, Omega_L):
        return 1.0 / np.sqrt(Omega_m * (1 + z_prime)**3 + Omega_L)
    integral, _ = quad(integrand, 0, z, args=(Omega_m, Omega_L))
    return (c * (1 + z) / H0) * integral

# Vectorize the luminosity distance function
def luminosity_distance(z, H0, Omega_m, Omega_L):
    return np.array([luminosity_distance_single(z_i, H0, Omega_m, Omega_L) for z_i in z])

# Distance modulus function
def distance_modulus(z, H0, Omega_m, Omega_L):
    d_L = luminosity_distance(z, H0, Omega_m, Omega_L)  # in Mpc
    return 5 * np.log10(d_L) + 25


# Initial guesses for parameters: H0, Omega_m, Omega_L
p0 = [70, 0.3, 0.7]

# Perform fit
params, covariance = curve_fit(
    lambda z, H0, Omega_m, Omega_L: distance_modulus(z, H0, Omega_m, Omega_L),
    redshifts,
    dms,
    p0=p0
)

# Best-fit parameters
H0_fit, Omega_m_fit, Omega_L_fit = params
print(f'H0 = {H0_fit}, Omega_m = {Omega_m_fit}, Omega_L = {Omega_L_fit}')

# Generate the fit curve
z_fit = np.linspace(min(redshifts), max(redshifts), 500)
mu_fit = distance_modulus(z_fit, H0_fit, Omega_m_fit, Omega_L_fit)

# Plot
plt.scatter(redshifts, dms, label='Data')
plt.plot(z_fit, mu_fit, color='red', label='Best-fit curve')
plt.xlabel('Redshift (z)')
plt.ylabel('Distance Modulus (mu)')
plt.legend()
plt.show()
