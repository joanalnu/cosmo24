# importing libraries
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from astropy.cosmology import FlatLambdaCDM, LambdaCDM
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
df = pd.read_csv(f'{data_path}/Results_SNCOSMO.csv')
df = df.dropna()

# applying cuts for clear outliners
# df = df[df['dm']>0]
df = df[df['dm']<40]
# df = df[df['dmerr']>0]
df = df[df['dmerr']<100] # avoiding one extreme errorbar
df = df[df['redshift']<0.125]

# additional cuts # new cuts, not the same as in original hubble_diagram.py
# df = df.drop(df[(df['redshift'] > 0.05) & (df['dm'] < 34)].index)
# df = df.drop(df[(df['redshift'] > 0.09) & (df['dm'] > 36)].index)

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





# Computing fits

# Define cosmological models
def model_68_lambdacdm(x_data, Om0, Ode0):
    cosmo = LambdaCDM(H0=68, Om0=Om0, Ode0=Ode0)
    return cosmo.distmod(x_data).value

def model_70_lambdacdm(x_data, Om0, Ode0):
    cosmo = LambdaCDM(H0=70, Om0=Om0, Ode0=Ode0)
    return cosmo.distmod(x_data).value

def model_72_lambdacdm(x_data, Om0, Ode0):
    cosmo = LambdaCDM(H0=72, Om0=Om0, Ode0=Ode0)
    return cosmo.distmod(x_data).value

def model_70_flatlambdacdm(x_data, Om0):
    cosmo = FlatLambdaCDM(H0=70, Om0=Om0)
    return cosmo.distmod(x_data).value

# Perform cosmological fits
cosmo_guess = [0.3, 0.7]
cosmo_bounds = ([0.1, 0.5], [0.5, 0.9])
popt_68, pcov_68 = curve_fit(model_68_lambdacdm, redshifts, dms, sigma=dm_errs, p0=cosmo_guess, bounds=cosmo_bounds)
popt_70, pcov_70 = curve_fit(model_70_lambdacdm, redshifts, dms, sigma=dm_errs, p0=cosmo_guess, bounds=cosmo_bounds)
popt_72, pcov_72 = curve_fit(model_72_lambdacdm, redshifts, dms, sigma=dm_errs, p0=cosmo_guess, bounds=cosmo_bounds)
popt_70flat, pcov_70flat = curve_fit(model_70_flatlambdacdm, redshifts, dms, sigma=dm_errs, p0=[0.3], bounds=([0.1], [0.5]))

Om0_68, Ode0_68 = popt_68
Om0_70, Ode0_70 = popt_70
Om0_72, Ode0_72 = popt_72
Om0_70flat = popt_70flat
std_error_68 = np.sqrt(np.diag(pcov_68))
std_error_70 = np.sqrt(np.diag(pcov_70))
std_error_72 = np.sqrt(np.diag(pcov_72))
std_error_70flat = np.sqrt(np.diag(pcov_70flat))

print("Om0_68: ", Om0_68, " Ode0_68: ", Ode0_68)
print("Om0_70: ", Om0_70, " Ode0_70: ", Ode0_70)
print("Om0_72: ", Om0_72, " Ode0_72: ", Ode0_72)
print("Om0_70flat: ", Om0_70flat)


# Generate redshift values for plotting
z_plot = np.linspace(0.001, 0.125, 1000)

# Compute best-fit cosmological models
distmod_68 = model_68_lambdacdm(z_plot, Om0_68, Ode0_68)
distmod_70 = model_70_lambdacdm(z_plot, Om0_70, Ode0_70)
distmod_72 = model_72_lambdacdm(z_plot, Om0_72, Ode0_72)
distmod_70flat = model_70_flatlambdacdm(z_plot, Om0_70flat)

# Plot results
# Plot results with residuals
fig, axs = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})

# Main plot
axs[0].scatter(redshifts, dms, color='blue')
axs[0].errorbar(redshifts, dms, yerr=dm_errs, color='blue', fmt='o')
axs[0].plot(z_plot, distmod_68, label=f"Best Fit (\u03A9m0 = {Om0_68:.2f}, \u03A9de0 = {Ode0_68:.2f})", color='red')
axs[0].plot(z_plot, distmod_70, label=f"Best Fit (\u03A9m0 = {Om0_70:.2f}, \u03A9de0 = {Ode0_70:.2f})", color='green')
axs[0].plot(z_plot, distmod_72, label=f"Best Fit (\u03A9m0 = {Om0_72:.2f}, \u03A9de0 = {Ode0_72:.2f})", color='purple')
axs[0].plot(z_plot, distmod_70flat, label=f"Best Fit (\u03A9m0 = {Om0_70flat:.2f})", color='orange')
axs[0].set_xlabel("Redshift (z)")
axs[0].set_ylabel("Distance Modulus (\u03bc)")
axs[0].legend()
axs[0].set_title("Cosmological Fits")

# Calculate residuals
residuals_68 = dms - model_68_lambdacdm(redshifts, Om0_68, Ode0_68)
residuals_70 = dms - model_70_lambdacdm(redshifts, Om0_70, Ode0_70)
residuals_72 = dms - model_72_lambdacdm(redshifts, Om0_72, Ode0_72)
residuals_70flat = dms - model_70_flatlambdacdm(redshifts, Om0_70flat)

# Standard deviation of residuals for each fit
std_dev_flat = np.std(residuals_68)
std_dev_lam = np.std(residuals_70)
std_dev_flat_lam = np.std(residuals_72)
std_dev_flat_70 = np.std(residuals_70flat)

# Print standard deviations
print(f"Standard Deviation for Flat LambdaCDM: {std_dev_flat:.4f}")
print(f"Standard Deviation for LambdaCDM: {std_dev_lam:.4f}")
print(f"Standard Deviation for LambdaCDM (72): {std_dev_flat_lam:.4f}")
print(f"Standard Deviation for Flat LambdaCDM (70): {std_dev_flat_70:.4f}")



# Residuals plot
axs[1].scatter(redshifts, residuals_68, color='red', label=f"Best Fit (\u03A9m0 = {Om0_68:.2f}, \u03A9de0 = {Ode0_68:.2f})")
axs[1].scatter(redshifts, residuals_70, color='green', label=f"Best Fit (\u03A9m0 = {Om0_70:.2f}, \u03A9de0 = {Ode0_70:.2f})")
axs[1].scatter(redshifts, residuals_72, color='purple', label=f"Best Fit (\u03A9m0 = {Om0_72:.2f}, \u03A9de0 = {Ode0_72:.2f})")
axs[1].scatter(redshifts, residuals_70flat, color='orange', label=f"Best Fit (\u03A9m0 = {Om0_70flat:.2f})")
axs[1].hlines(0, xmin=min(redshifts), xmax=max(redshifts), colors='gray', linestyles='dashed')
axs[1].set_xlabel("Redshift (z)")
axs[1].set_ylabel("Residuals")
axs[1].legend()
axs[1].set_title("Residuals")

plt.tight_layout()
plt.show()
fig.savefig(f'{dirpath}/hubble_diagram.png')





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


# Hubble constant values to fix
H0_values = [67, 70, 73]

# Store fit results
fit_results = []
z_fit = np.linspace(min(redshifts), max(redshifts), 500)

plt.figure(figsize=(10, 6))

for H0_fixed in H0_values:
    # Perform fit with fixed H0
    def model(z, Omega_m, Omega_L):
        return distance_modulus(z, H0_fixed, Omega_m, Omega_L)

    # Initial guesses for Omega_m and Omega_L
    p0 = [0.3, 0.7]

    params, covariance = curve_fit(model, redshifts, dms, p0=p0, bounds = ([0.1, 0.5], [0.5, 0.9]))
    Omega_m_fit, Omega_L_fit = params
    
    # Generate the fit curve
    mu_fit = model(z_fit, Omega_m_fit, Omega_L_fit)

    # Save results
    fit_results.append((H0_fixed, Omega_m_fit, Omega_L_fit))

    # Plot the fit
    plt.plot(z_fit, mu_fit, label=f"$H_0$ = {H0_fixed}, $Omega_m$ = {Omega_m_fit:.2f}, $Omega_\Lambda$ = {Omega_L_fit:.2f}$")

# Plot data
plt.scatter(redshifts, dms, color='black', label='Data')
plt.xlabel('Redshift (z)')
plt.ylabel('Distance Modulus (mu)')
plt.legend()
plt.title('Fits with Fixed Hubble Constants')
plt.grid()
plt.show()

# Print fit results
for H0_fixed, Omega_m_fit, Omega_L_fit in fit_results:
    print(f"H0: {H0_fixed}, Omega_m: {Omega_m_fit:.2f}, Omega_L: {Omega_L_fit:.2f}")
