# initial settings
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad
from wiserep_api import get_target_property
dirpath = os.path.dirname(os.path.abspath(__file__))
sncosmo_path = dirpath + '/Results_SNCOSMO.csv'

def read_sn_data():
    """
    Read SN data from Results_SNCOSMO.csv file. This creates a csv file with the used data.
    The function returns the names, redshifts, distance moduli, distance modulus errors, velocities, distances and distance errors.
    """
    # Read SN data
    df = pd.read_csv(sncosmo_path)
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
    
    hosts = list()
    types = list()
    coords = list()
    for name in names:
        hosts.append(get_target_property(name, 'host'))
        types.append(get_target_property(name, 'type'))
        coords.append(get_target_property(name, 'coords'))

    df_used = pd.DataFrame({'name':names, 'redshift':redshifts, 'dm':dms, 'dmerr':dm_errs, 'host':hosts, 'type':types, 'coordinates':coords})

    df_used.to_csv(f'{dirpath}/used_data.csv', index=False)

    return names, redshifts, dms, dm_errs, velocities, distances, sigma_distances


# READ SN data
names, redshifts, dms, dm_errs, velocities, distances, sigma_distances = read_sn_data()


# Plot Hubble diagram

# 0 - Compute fits
from astropy.cosmology import LambdaCDM, FlatLambdaCDM
z = np.linspace(0.001, 0.125, 1000)

## 0.1 - LambdaCDM fit
# TODO: find what best fits the data
model_67_lambdacdm = LambdaCDM(H0=67.4, Om0=0.3, Ode0=0.7) # Planck results
model_67_lambdacdm_down = LambdaCDM(H0=66.9, Om0=0.3, Ode0=0.7) # Planck results
model_67_lambdacdm_up = LambdaCDM(H0=67.9, Om0=0.3, Ode0=0.7) # Planck results
# pm 0.5

model_70_lambdacdm = LambdaCDM(H0=69.96, Om0=0.3, Ode0=0.7) # CCHP program results (Freedman et al. 2024)
model_70_lambdacdm_down = LambdaCDM(H0=68.92, Om0=0.3, Ode0=0.7)
model_70_lambdacdm_up = LambdaCDM(H0=71.0, Om0=0.3, Ode0=0.7)
# pm 1.05

model_73_lambdacdm = LambdaCDM(H0=73, Om0=0.3, Ode0=0.7) # SH0ES collaboration results (Riess et al. 2022)
model_73_lambdacdm_down = LambdaCDM(H0=71.95, Om0=0.3, Ode0=0.7)
model_73_lambdacdm_up = LambdaCDM(H0=74.04, Om0=0.3, Ode0=0.7)
# pm 1.04

distmod_67 = model_67_lambdacdm.distmod(z).value
distmod_67_down = model_67_lambdacdm_down.distmod(z).value
distmod_67_up = model_67_lambdacdm_up.distmod(z).value
distmod_70 = model_70_lambdacdm.distmod(z).value
distmod_70_down = model_70_lambdacdm_down.distmod(z).value
distmod_70_up = model_70_lambdacdm_up.distmod(z).value
distmod_73 = model_73_lambdacdm.distmod(z).value
distmod_73_down = model_73_lambdacdm_down.distmod(z).value
distmod_73_up = model_73_lambdacdm_up.distmod(z).value

## 0.2 - FlatLambdaCDM fit
model_67_flatlambdacdm = FlatLambdaCDM(H0=67.4, Om0=0.3) # Planck results
model_67_flatlambdacdm_down = FlatLambdaCDM(H0=66.9, Om0=0.3) # Planck results
model_67_flatlambdacdm_up = FlatLambdaCDM(H0=67.9, Om0=0.3) # Planck results

model_70_flatlambdacdm = FlatLambdaCDM(H0=69.96, Om0=0.3) # CCHP program results (Freedman et al. 2024)
model_70_flatlambdacdm_down = FlatLambdaCDM(H0=68.92, Om0=0.3)
model_70_flatlambdacdm_up = FlatLambdaCDM(H0=71.0, Om0=0.3)

model_73_flatlambdacdm = FlatLambdaCDM(H0=73, Om0=0.3) # SH0ES collaboration results (Riess et al. 2022)
model_73_flatlambdacdm_down = FlatLambdaCDM(H0=71.95, Om0=0.3)
model_73_flatlambdacdm_up = FlatLambdaCDM(H0=74.04, Om0=0.3)

distmod_67_flat = model_67_flatlambdacdm.distmod(z).value
distmod_67_flat_down = model_67_flatlambdacdm_down.distmod(z).value
distmod_67_flat_up = model_67_flatlambdacdm_up.distmod(z).value
distmod_70_flat = model_70_flatlambdacdm.distmod(z).value
distmod_70_flat_down = model_70_flatlambdacdm_down.distmod(z).value
distmod_70_flat_up = model_70_flatlambdacdm_up.distmod(z).value
distmod_73_flat = model_73_flatlambdacdm.distmod(z).value
distmod_73_flat_down = model_73_flatlambdacdm_down.distmod(z).value
distmod_73_flat_up = model_73_flatlambdacdm_up.distmod(z).value

## 0.3 - Luminosity distance fit
def luminosity_distance(z, H0, Omega_m, Omega_L):
    def luminosity_distance_single(z, H0, Omega_m, Omega_L):
        def integrand(z_prime, Omega_m, Omega_L):
            return 1/np.sqrt(Omega_m*(1+z_prime)**3 + Omega_L)# + (1 - Omega_m - Omega_L)*(1+z_prime)**2)
        integral, _ = quad(integrand, 0, z, args=(Omega_m, Omega_L))
        return (299792.458 * (1 + z) / H0) * integral
    return np.array([luminosity_distance_single(z_i, H0, Omega_m, Omega_L) for z_i in z])

def distance_modulus(z, H0, Omega_m, Omega_L):
    d_L = luminosity_distance(z, H0, Omega_m, Omega_L) # in Mpc
    return 5 * np.log10(d_L) + 25

def model(z, H0, Omega_m, Omega_L):
    return distance_modulus(z, H0, Omega_m, Omega_L)

p0 = [70, 0.3, 0.7] # guess
bounds = ([0.0, 0.0, 0.0],[1000, 1000, 1000])# bounds
params, cov = curve_fit(model, redshifts, dms, p0=p0, sigma=dm_errs)
# params, cov = curve_fit(model, redshifts, dms, p0=p0, sigma=dm_errs, bounds=bounds)
H0_fit, Omega_m_fit, Omega_L_fit = params
H0_fit_err, Omega_m_fit_err, Omega_L_fit_err = np.sqrt(np.diag(cov))
mu_fit = model(z, H0_fit, Omega_m_fit, Omega_L_fit) # generate fit curve (equivalent to distmod)


## 0.4 - Alternative universes
model_accelerating = LambdaCDM(H0=69, Om0=2.0, Ode0=2.0) # accelerating universe
model_decelerating_closed = LambdaCDM(H0=69, Om0=2.5, Ode0=-1.0) # decelerating universe, closed
model_decelerating_open = LambdaCDM(H0=69, Om0=0.5, Ode0=-1.0) # decelerating universe, open

distmod_accelerating = model_accelerating.distmod(z).value
distmod_decelerating_closed = model_decelerating_closed.distmod(z).value
distmod_decelerating_open = model_decelerating_open.distmod(z).value



# 1 - Plot SN data with all different fits and residuals for the LambdaCDM fits
fig, ax = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})

# main plot
ax[0].errorbar(redshifts, dms, yerr=dm_errs, fmt='o', color='blue', label='SNCOSMO results')

# LambdaCDM fits
ax[0].plot(z, distmod_67, color='black', label='LambdaCDM fit (H0=67.4, Om0=0.3, Ode0=0.7)')
ax[0].fill_between(z, distmod_67_down, distmod_67_up, color='black', alpha=0.3)
ax[0].plot(z, distmod_70, color='red', label='LambdaCDM fit (H0=69.96, Om0=0.3, Ode0=0.7)')
ax[0].fill_between(z, distmod_70_down, distmod_70_up, color='red', alpha=0.3)
ax[0].plot(z, distmod_73, color='green', label='LambdaCDM fit (H0=73, Om0=0.3, Ode0=0.7)')
ax[0].fill_between(z, distmod_73_down, distmod_73_up, color='green', alpha=0.3)

# FlatLambdaCDM fits
# ax[0].plot(z, distmod_67_flat, color='black', label='LambdaCDM fit (H0=67.4, Om0=0.3)', linestyle='--')
# ax[0].fill_between(z, distmod_67_down, distmod_67_up, color='black', alpha=0.3)
# ax[0].plot(z, distmod_70_flat, color='red', label='LambdaCDM fit (H0=69.96, Om0=0.3)', linestyle='--')
# ax[0].fill_between(z, distmod_70_down, distmod_70_up, color='red', alpha=0.3)
# ax[0].plot(z, distmod_73_flat, color='green', label='LambdaCDM fit (H0=73, Om0=0.3)', linestyle='--')
# ax[0].fill_between(z, distmod_73_down, distmod_73_up, color='green', alpha=0.3)

# luminosity distance fit
ax[0].plot(z, mu_fit, label=f'Luminosity distance fit (H0={H0_fit:.2f}, Om0={Omega_m_fit:.2f}, Ode0={Omega_L_fit:.2f})', color='pink')

# alternative universes
# ax[0].plot(z, distmod_accelerating, color='orange', label='Accelerating universe', linestyle='-.') # not needed (LambdaCDM already represents this)
ax[0].plot(z, distmod_decelerating_closed, color='orange', label='Decelerating universe (closed)', linestyle='-.')
# ax[0].plot(z, distmod_decelerating_open, color='lightblue', label='Decelerating universe (open)', linestyle='-.') # does not appear in plot


# Residuals plot
## compute residuals
residuals_67 = dms - model_67_lambdacdm.distmod(redshifts).value
residuals_70 = dms - model_70_lambdacdm.distmod(redshifts).value
residuals_73 = dms - model_73_lambdacdm.distmod(redshifts).value
residuals_fit = dms - model(redshifts, H0_fit, Omega_m_fit, Omega_L_fit)

std_dev_67 = np.std(residuals_67)
std_dev_70 = np.std(residuals_70)
std_dev_73 = np.std(residuals_73)
std_dev_fit = np.std(residuals_fit)

## plot residuals
ax[1].hlines(0, 0, 0.125, color="pink", linestyle='--')
# ax[1].errorbar(redshifts, residuals_67, yerr=dm_errs, fmt='o', color='black', label=f'H0=67.4, $\sigma$={std_dev_67:.2f}')
# ax[1].errorbar(redshifts, residuals_70, yerr=dm_errs, fmt='o', color='red', label=f'H0=69.96, $\sigma$={std_dev_70:.2f}')
# ax[1].errorbar(redshifts, residuals_73, yerr=dm_errs, fmt='o', color='green', label=f'H0=73, $\sigma$={std_dev_73:.2f}')
ax[1].errorbar(redshifts, residuals_fit, yerr=dm_errs, fmt='o', color='blue', label=f'Fit, $\sigma$={std_dev_fit:.2f}')



# Plot formailities
ax[0].set_xlabel('$z$')
ax[0].set_ylabel('$\mu$')
ax[0].legend()
ax[0].grid()

ax[1].set_xlabel('$z$')
ax[1].set_ylabel('$\mu$')
ax[1].legend()
ax[1].grid()

ax[0].set_xlim(0, 0.125)
ax[0].set_ylim(30, 45)
plt.title('Hubble diagram')
plt.show()
plt.tight_layout()
fig.savefig(f'{dirpath}/hubble_diagram.png')

