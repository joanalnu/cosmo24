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
df = pd.read_csv(f'{data_path}/Results_SNCOSMO.csv')
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
# define linear function
def model(x, a, b):
    return a * x + b

# ACTUALLY THIS IS NOT USED; INSTEAD IT IS DONE WITH THE ASTROPY.COSMO MODEL
# computing SN dm-redshift fit
popt, pcov = curve_fit(model, redshifts, dms)
slope, intercept = popt
best_fit = slope * redshifts + intercept

# computing SN distances-velocities fit
popt2, pcov2 = curve_fit(model, velocities, distances)
slope2, intercept2 = popt2
best_fit2 = slope2 * velocities + intercept2

# computing cepheid fit (linear)
full_vels = np.concatenate((vel2, vel2))
full_dists = np.concatenate((v_dist, i_dist))
full_errs = np.concatenate((v_dist_err, i_dist_err))
popt4, pcov4 = curve_fit(model, full_vels, full_dists)
slope4, intercept4 = popt4
cepheid_fit = slope4 * full_vels + intercept4

# compute cepheid-sn fit (velocity-distances, linear)
merged_vels = np.concatenate((velocities, vel2, vel2))
merged_dist = np.concatenate((distances, v_dist, i_dist))
merged_errs = np.concatenate((sigma_distances, v_dist_err, i_dist_err))
popt3, pcov3 = curve_fit(model, merged_vels, merged_dist)
slope3, intercept3 = popt3
best_fit3 = slope3 * merged_vels + intercept3

# Compute Ho uncertaininty
# SN (slope2)
residuals = distances - best_fit2 # Compute residuals
mse = np.sum(residuals**2) / (len(distances) - 2) # Compute mean squared error (MSE)
variance = np.sum((distances - np.mean(distances))**2) # Compute variance of distances
std_error2 = np.sqrt(mse / variance) # Compute standard error of the slope

# Cepheid (Slope4)
residuals = full_dists - cepheid_fit
# weighted_residuals = residuals / full_errs
mse = np.sum(residuals**2) / (len(full_vels)-2)
variance = np.sum((full_dists - np.mean(full_dists))**2)
std_error4 = np.sqrt(mse/variance)

# Sn + Cepheid (slope3)
residuals = merged_dist - best_fit3
# weighted_residuals = residuals / merged_errs
mse = np.sum(residuals**2) / (len(merged_vels)-2)
variance = np.sum((merged_dist - np.mean(merged_dist))**2)
std_error3 = np.sqrt(mse/variance)





# Compute fits with COSMOLOGY
from astropy.cosmology import FlatLambdaCDM, LambdaCDM

# define models
cosmo_org = FlatLambdaCDM(H0=70, Om0=0.3)
cosmo_manual = FlatLambdaCDM(H0=70, Om0=0.9)
cosmo2 = LambdaCDM(H0=70, Om0=0.3, Ode0=1.7)
cosmo3 = LambdaCDM(H0=50, Om0=0.3, Ode0=0.0)
def model_flatlambdacdm(x_data, H0, Om0):
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    distmod = cosmo.distmod(x_data).value
    return distmod

def model_lambdacdm(x_data, H0, Om0, Ode0):
    cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)
    distmod = cosmo.distmod(x_data).value
    return distmod

def model_computeHo(x_data, H0):
    cosmo = FlatLambdaCDM(H0=H0, Om0=0.3)
    distmod = cosmo.distmod(x_data).value
    return distmod

# compute best fit models
# model_flatlambdacdm
guess = [70, 0.3]
bounds = ([67.0, 0.0],[73.0, 1.0])
popt_flat, pcov_flat = curve_fit(model_flatlambdacdm, redshifts, dms, sigma=dm_errs, p0=guess, bounds=bounds)
H0_best_flat, Om0_best_flat = popt_flat
H0_best_flat_err, Om0_best_flat_err = np.sqrt(np.diag(pcov_flat)) # std dev of parameters
print(f'Astropycosmo fit flat:\t H0 = {H0_best_flat} ± {H0_best_flat_err}\n\t\t\t Om0 = {Om0_best_flat} ± {Om0_best_flat_err}\n')

# model_lambdacdm
guess = [70, 0.3, 0.7]
bounds = ([67.0, 0.0, 0.0],[73.0, 1.0, 1.0])
popt_lam, pcov_lam = curve_fit(model_lambdacdm, redshifts, dms, sigma=dm_errs, p0=guess, bounds=bounds)
H0_best_lam, Om0_best_lam, Ode0_best_lam = popt_lam
H0_best_lam_err, Om0_best_lam_err, Ode0_best_lam_err = np.sqrt(np.diag(pcov_lam)) # std dev of parameters
print(f'Astropycosmo fit noflat: H0 = {H0_best_lam} ± {H0_best_lam_err}\n\t\t\t Om0 = {Om0_best_lam} ± {Om0_best_lam_err}\n\t\t\t Ode0 = {Ode0_best_lam} ± {Ode0_best_lam_err}\n')

guess = [70, 0.3, 0.7]
# bounds = ([67.0, 0.2, 0.7],[73.0, 0.4, 0.8]) # restrictive
bounds = ([0.0,0.0,0.0],[1000.0, 1000.0, 1000.0]) # free
# popt_res, pcov_res = curve_fit(model_lambdacdm, redshifts, dms, sigma=dm_errs, p0=guess, bounds=bounds)
# H0_res, Om0_res, Ode0_res = popt_res
# H0_best_res_err, Om0_best_res_err = np.sqrt(np.diag(pcov_res)) # std dev of parameters
# print(f'Astropycosmo fit res: H0 = {H0_best_res} ± {H0_best_res_err}\n\t\t\tOm0 = {Om0_best_res} ± {Om0_best_res_err}\n')

# model_computeHo
guess = 70
# bounds = []
popt_cpt, pcov_cpt = curve_fit(model_computeHo, redshifts, dms, sigma=dm_errs, p0=guess)
H0_best_cpt = float(popt_cpt)
H0_best_cpt = float(H0_best_cpt)
H0_best_cpt_err = np.sqrt(np.diag(pcov_cpt)) # std dev of parameters
print(f'Astropycosmo fit onlyHo: H0 = {H0_best_cpt} ± {H0_best_cpt_err}\n\t\t\t Om0 = 0.3 ± 0.0\n')


# define redshift array (z axis)
z=np.linspace(0.001, 0.17, 1000)

# Compute distmods
distmod_manual = cosmo_manual.distmod(z).value
distmod2 = cosmo2.distmod(z).value
distmod_org = cosmo_org.distmod(z).value
distmod3 = cosmo3.distmod(z).value
distmod_bestfit_flat = model_flatlambdacdm(z, H0_best_flat, Om0_best_flat)
distmod_bestfit_lam = model_lambdacdm(z, H0_best_lam, Om0_best_lam, Ode0_best_lam)
# distmod_bestfit_res = model_lambdacdm(z, H0_res, Om0_res, Ode0_res)
distmod_bestfit_cpt = model_computeHo(z, H0_best_cpt)

# plot cosmological models
plt.figure()
plt.scatter(redshifts, dms)
plt.errorbar(redshifts, dms, yerr=[dm_errs], fmt='none')

plt.plot(z, distmod2, label='distmod2', color=(187/255,58/255,50/255))
plt.plot(z, distmod3, label='distmod3', color=(81/255,197/255,58/255))
plt.plot(z, distmod_org, label='original', color='gray', alpha=0.8)
plt.plot(z, distmod_bestfit_cpt, label=f'Best Fit (only $H_0$): $H_0$={H0_best_cpt:.5f} ($\Omega_m$=0.3)')
plt.plot(z, distmod_manual, label='distmod', color=(239/255, 134/255, 54/255))
plt.plot(z, distmod_bestfit_flat, label=f'Best Fit: $H_0$={H0_best_flat:.5f}, $\Omega_m$={Om0_best_flat:.5f}', color='red', linewidth=1)
plt.plot(z, distmod_bestfit_lam, label=f'Best Fit: $H_0$={H0_best_lam:.5f}, $\Omega_m$={Om0_best_lam:.5f}, $\Omega_\Lambda$={Ode0_best_lam:.5f}', color='orange', linewidth=1, linestyle='dashed')
# plt.plot(z, distmod_bestfit_res, label=f'Best Fit: $H_0$={H0_res:.5f}, $\Omega_m$={Om0_res:.5f}, $\Omega_\Lambda$={Ode0_res:.5f}', color='green', linewidth=1, linestyle='dashed')

plt.title('Cosmological models')
plt.legend()
plt.xlabel('$z$')
plt.ylabel('$\mu$')
plt.savefig('actual_fig.png', dpi=300)

# plot difference between models
diff = distmod_bestfit_flat - distmod_bestfit_lam # change your preferences
plt.figure()
plt.plot(z, diff)
plt.savefig('diff.png')






##################################################################################################################################################################
# otras cositas
cosmo_best = LambdaCDM(H0=H0_best_lam, Om0=Om0_best_lam, Ode0=Ode0_best_lam, Tcmb0=2.7)
z_evo = np.linspace(0.001, 1.0, num=100)
h_z = [cosmo_best.H(red).value for red in z_evo]
Om_z = [cosmo_best.Om(red) for red in z_evo]
Ode_z = [cosmo_best.Ode(red) for red in z_evo]
Ogamma_z = [cosmo_best.Ogamma(red) for red in z_evo]
Tcmb_z = [cosmo_best.Tcmb(red).value for red in z_evo]
random_fig, axs = plt.subplots(3, 2)
axs[0,0].plot(z_evo, h_z, label='$H(z)$')
axs[0,1].plot(z_evo, Om_z, label='$\Omega_m$')
axs[1,0].plot(z_evo, Ode_z, label='$\Omega_\Lambda$')
axs[1,1].plot(z_evo, Ogamma_z, label='$\Omega_\gamma$')
axs[2,1].plot(z_evo, Tcmb_z, label='T CMB')
axs[0,0].legend()
axs[0,1].legend()
axs[1,0].legend()
axs[1,1].legend()
axs[2,1].legend()
# axs[2,2].plot(z_evo, h_z, label='$H(z)$')
# axs[2,2].plot(z_evo, Om_z, label='$\Omega_m$')
# axs[2,2].plot(z_evo, Ode_z, label='$\Omega_\Lambda$')
# axs[2,2].plot(z_evo, Ogamma_z, label='$\Omega_\gamma$')
# axs[2,2].plot(z_evo, Tcmb_z, label='T CMB')
plt.tight_layout()
plt.legend()
random_fig.savefig('max_randomness.png',dpi=300)
###################################################################################################################################################################################################################################################



# plotting
fig, axs = plt.subplots(2, 2, figsize=(10,8))
# SN only redshifts-dms (cosmo fit)
axs[0,0].scatter(redshifts, dms)
axs[0,0].errorbar(redshifts, dms, yerr=[dm_errs], fmt='None')
axs[0,0].plot(z, distmod_manual, label='distmod', color='r') # distmod_manual model
axs[0,0].set_xlabel('Redshift z')
axs[0,0].set_ylabel('Distance modulus µ')
axs[0,0].set_title('SN only')
axs[0,0].legend()

# SN only velcoities-distances (linear fit)
axs[0,1].scatter(velocities, distances, marker="2")
axs[0,1].errorbar(velocities, distances, yerr=[sigma_distances], fmt='None')
axs[0,1].plot(velocities, best_fit2, color='r', label=f'Ho(SN) {(1/slope2):.3f}±{std_error2:.3f}') #best_fit2 model
axs[0,1].set_xlabel('Velocity [km/s]')
axs[0,1].set_ylabel('Distances [Mpc]')
axs[0,1].set_title('SN only')
axs[0,1].legend()

# Cepheid only velocities-distances (linear fit)
axs[1,0].scatter(vel2, i_dist, label='I Cepheid', marker="2")
axs[1,0].errorbar(vel2, i_dist, yerr=[i_dist_err], fmt='None')
axs[1,0].scatter(vel2, v_dist, label='V Cepheid', marker="2")
axs[1,0].errorbar(vel2, v_dist, yerr=[v_dist_err], fmt='None')
# axs[1,0].plot(full_vels, cepheid_fit, color=(0.0,1.0,0.0,0.5), label=f'Ho(SN) {(1/slope4):.3f}±{std_error4:.3f}') # cepheid_fit model
axs[1,0].set_xlabel('Velocity [km/s]')
axs[1,0].set_ylabel('Distances [Mpc]')
axs[1,0].set_title('Cepheid Only')
axs[1,0].legend()

# Cepheid-SN velcities-distances (linear fit)
# SN
axs[1,1].scatter(velocities, distances, label='SN', marker="2")
axs[1,1].errorbar(velocities, distances, yerr=[sigma_distances], fmt='None')
axs[1,1].plot(velocities, best_fit2, color=(1.0, 0.0, 0.0, 0.5), label=f'Ho(SN) {(1/slope2):.3f}±{std_error2:.3f}') # best_fit2 model
# Cepheid
axs[1,1].scatter(vel2, i_dist, label='I Cepheid', marker="2")
axs[1,1].errorbar(vel2, i_dist, yerr=[i_dist_err], fmt='None')
axs[1,1].scatter(vel2, v_dist, label='V Cepheid', marker="2")
axs[1,1].errorbar(vel2, v_dist, yerr=[v_dist_err], fmt='None')
axs[1,1].plot(full_vels, cepheid_fit, color=(0.0,1.0,0.0,0.5), label=f'Ho(SN) {(1/slope4):.3f}±{std_error4:.3f}') # cepheid_fit model
# merged fit
axs[1,1].plot(merged_vels, best_fit3, color=(1.0,0.0, 0.0, 1.0), label=f'Ho(SN) {(1/slope3):.3f}±{std_error3:.3f}') # best_fit3 model
axs[1,1].set_xlabel('velocities [km/s]')
axs[1,1].set_ylabel('distance')
axs[1,1].set_title('cepheid + SN')
axs[1,1].legend()

plt.tight_layout()
fig.savefig(f'{dirpath}/ultimate-fig.png', dpi=512)


# # plotting SN + cepheid velociti-distance (linear fits) alone
fig3, ax = plt.subplots(1,1, figsize=(6,6))
ax.scatter(velocities, distances, label='SN', color=to_rgb(197,58,50), marker='*')
ax.errorbar(velocities, distances, yerr=[sigma_distances], fmt='None', color='gray')
ax.plot(velocities, best_fit2, color='r', label=f'Only SNIa Ho: {(1/slope2):.3f}±{std_error2:.3f}', alpha=0.5)
ax.scatter(vel2, v_dist, label='V Cepheids', color=to_rgb(81,157,61), marker='2')
ax.errorbar(vel2, v_dist, yerr=[v_dist_err], fmt='None', color='gray')
ax.scatter(vel2, i_dist, label='I Cepheids', color=to_rgb(238,136,47), marker='2')
ax.errorbar(vel2, i_dist, yerr=[i_dist_err], fmt='None', color='gray')
ax.plot(full_vels, cepheid_fit, color=(0.0,1.0,0.0,1.0), label=f'Cepheid fit Ho: {(1/slope4):.3f}±{std_error4:.3f}')
ax.plot(merged_vels, best_fit3, color='r', label=f'Combined Ho: {(1/slope3):.3f}±{std_error3:.3f}')
ax.set_xlabel('Velocities [km/s]')
ax.set_ylabel('Distances [Mpc]')
ax.set_title('Hubble Diagram')
ax.legend()
fig3.savefig('mostbeautifulfigureinuniverse.png', dpi=600)

# Compute hubble time with linear fit (best_fit2 model)
Ho_SN = (1/slope2)
Ho_SN = Ho_SN * (1/(3.086*(10**19))) # Km/s/Mpc -> 1/s 1 Mpc= 3.086 * 10**19 km
t_SN = 1/Ho_SN # compute t_Ho
t_SN = t_SN / (86400*365.25*(10**9)) # s -> Gyr
sigma_t_SN = t_SN* (std_error2/Ho_SN)

print()
print(f'Nº of SN {len(velocities)}')
print(f'SNIa-based Ho (linear):  {(1/slope2)} ± {std_error2}')
print()
print(f'SNIa-based tH (linear):  {t_SN} ± {sigma_t_SN}')
print()
print()
print(f'Nº of Cepheid-Galaxies:  {len(v_dist)}')
print(f'Cepheid-based Ho (linear): {(1/slope4)} ± {std_error4}')
print()
print(f'Combined Ho fit (linear): {1/slope3} ± {std_error3}')