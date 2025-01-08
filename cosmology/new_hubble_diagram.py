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
df_used = pd.DataFrame({'name':names, 'redshift':redshifts, 'dm':dms, 'dmerr':dm_errs})
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
def model_flatlambdacdm(x_data, H0, Om0):
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    return cosmo.distmod(x_data).value

def model_lambdacdm(x_data, H0, Om0, Ode0):
    cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)
    return cosmo.distmod(x_data).value

def model_computeHo(x_data, H0):
    cosmo = FlatLambdaCDM(H0=H0, Om0=0.3)
    return cosmo.distmod(x_data).value

# Perform cosmological fits
cosmo_guess_flat = [70, 0.3]
cosmo_bounds_flat = ([67.0, 0.0], [73.0, 1.0])
popt_flat, pcov_flat = curve_fit(model_flatlambdacdm, redshifts, dms, 
                                 sigma=dm_errs, p0=cosmo_guess_flat, bounds=cosmo_bounds_flat)
H0_flat, Om0_flat = popt_flat
std_error_flat = np.sqrt(np.diag(pcov_flat))

cosmo_guess_lam = [70, 0.3, 0.7]
cosmo_bounds_lam = ([67.0, 0.0, 0.0], [73.0, 1.0, 1.0])
popt_lam, pcov_lam = curve_fit(model_lambdacdm, redshifts, dms, 
                               sigma=dm_errs, p0=cosmo_guess_lam, bounds=cosmo_bounds_lam)
H0_lam, Om0_lam, Ode0_lam = popt_lam
std_error_lam = np.sqrt(np.diag(pcov_lam))

cosmo_guess_H0 = [70]
popt_H0, pcov_H0 = curve_fit(model_computeHo, redshifts, dms, sigma=dm_errs, p0=cosmo_guess_H0)
H0_only, = popt_H0
std_error_H0 = np.sqrt(np.diag(pcov_H0))

# Print fit results
print(f"Flat LambdaCDM Fit: H0 = {H0_flat:.5f} \u00b1 {std_error_flat[0]:.5f}, Om0 = {Om0_flat:.5f} \u00b1 {std_error_flat[1]:.5f}")
print(f"LambdaCDM Fit: H0 = {H0_lam:.5f} \u00b1 {std_error_lam[0]:.5f}, Om0 = {Om0_lam:.5f} \u00b1 {std_error_lam[1]:.5f}, Ode0 = {Ode0_lam:.5f} \u00b1 {std_error_lam[2]:.5f}")
print(f"Only H0 Fit: H0 = {H0_only:.5f} \u00b1 {std_error_H0[0]:.5f}")

# Generate redshift values for plotting
z_plot = np.linspace(0.001, 0.125, 1000)

# Compute best-fit cosmological models
distmod_flat = model_flatlambdacdm(z_plot, H0_flat, Om0_flat)
distmod_lam = model_lambdacdm(z_plot, H0_lam, Om0_lam, Ode0_lam)
distmod_H0 = model_computeHo(z_plot, H0_only)

# Plot results
# Plot results with residuals
fig, axs = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})

# Main plot
axs[0].scatter(redshifts, dms, label="SN Data", color='blue')
axs[0].errorbar(redshifts, dms, yerr=dm_errs, fmt="o", color='blue', label="SN Errors")
axs[0].plot(z_plot, distmod_flat, label=f"Flat LambdaCDM: H0 = {H0_flat:.2f}", color='r')
axs[0].plot(z_plot, distmod_lam, label=f"LambdaCDM: H0 = {H0_lam:.2f}", linestyle="--", color="g")
axs[0].plot(z_plot, distmod_H0, label=f"Only H0: H0 = {H0_only:.2f}", linestyle=":", color="b")
axs[0].set_xlabel("Redshift (z)")
axs[0].set_ylabel("Distance Modulus (\u03bc)")
axs[0].legend()
axs[0].set_title("Cosmological Fits")

# Residuals calculation
residuals_flat = dms - model_flatlambdacdm(redshifts, H0_flat, Om0_flat)
residuals_lam = dms - model_lambdacdm(redshifts, H0_lam, Om0_lam, Ode0_lam)
residuals_H0 = dms - model_computeHo(redshifts, H0_only)

# Residuals plot
axs[1].scatter(redshifts, residuals_flat, label="Flat LambdaCDM Residuals", color='r')
axs[1].scatter(redshifts, residuals_lam, label="LambdaCDM Residuals", color='g', alpha=0.6)
axs[1].scatter(redshifts, residuals_H0, label="Only H0 Residuals", color='b', alpha=0.6)
axs[1].hlines(0, xmin=min(redshifts), xmax=max(redshifts), colors='gray', linestyles='dashed')
axs[1].set_xlabel("Redshift (z)")
axs[1].set_ylabel("Residuals")
axs[1].legend()
axs[1].set_title("Residuals")

plt.tight_layout()
plt.show()

