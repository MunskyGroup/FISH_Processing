#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 00:00:00 2022

@author: luis_aguilera
"""

# Libraries
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gillespy2
import os
import pathlib
import seaborn as sns
import shutil
from tqdm.auto import tqdm
from functools import partial
from scipy.stats import multivariate_normal
import sys
from scipy.optimize import minimize

# Directory Setup
current_dir = pathlib.Path().absolute()
folder_outputs = current_dir.joinpath('Figures_Exercise')
#if os.path.exists(folder_outputs):
#    shutil.rmtree(folder_outputs)
folder_outputs.mkdir(parents=True, exist_ok=True)
# Plotting configuration
plt.rcParams.update({ 'axes.labelsize': 14, 'axes.titlesize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 10,})
colors = ['#FBD148', '#6BCB77', '#AA66CC', '#FF6B6B', '#4D96FF']
species_colors = {'G_off': colors[0], 'G_on': colors[1], 'R_n': colors[2], 'R_c': colors[3], 'P': colors[4]}

# function to calculate effective kt
def calculate_effective_kt(D, k_diff_r, transport_rate, model_type):
    D = D / 2
    if model_type == '2D':
        T_diff = D**2 / (4 * k_diff_r)
    elif model_type == '3D':
        T_diff = D**2 / (6 * k_diff_r)
    else:
        raise ValueError("Dimension must be 2 or 3.")
    T_transport = 1 / transport_rate
    T_total = T_diff + T_transport
    k_t = 1 / T_total
    return np.round(k_t,3)

# Loading parameters for MCMC
try:
    chain_length = int(sys.argv[1])
    burnin_time = 10000
except:
    chain_length = 10000
    burnin_time = int(chain_length/2) 
min_value_parameters = 0.0001
max_value_parameters = 100                          
run_ms = True
max_allowed_value_variables = 20000


# Parameters for the model
k_on = 0.5
k_off = 0.1
k_r = 3
k_p = 0.9
gamma_r = 0.05  # assuming gamma_rn = gamma_rc
gamma_p = 0.65
transport_rate = 1
diffusion_rate = 10  # assuming diffusion_rate_r = diffusion_rate_p
total_simulation_time = 201
nucleus_diameter = 40
number_of_trajectories = 50
cytosol_diameter = 70
model_type = '2D'
inhibition_constant = 0.1
drug_application_time = 120
kt = calculate_effective_kt(D=nucleus_diameter, k_diff_r=diffusion_rate, transport_rate=transport_rate, model_type=model_type)
print('Effective transport rate: ', kt)
max_y_val = 65
parameter_values = {'k_on': k_on, 'k_off': k_off, 'k_r': k_r, 'k_t': kt, 'k_p': k_p, 'gamma_rn': gamma_r, 'gamma_rc': gamma_r, 'gamma_p': gamma_p}
initial_conditions = {'G_off': 0, 'G_on': 1, 'R_n': 0, 'R_c': 0, 'P': 0}
# Updating parameter values
parameter_symbols = ['k_on', 'k_off', 'k_r', 'k_t', 'k_p', 'gamma_rn', 'gamma_rc', 'gamma_p', 'inhibition_constant']
true_parameter_values = [k_on, k_off, k_r, kt, k_p, gamma_r, gamma_r, gamma_p, inhibition_constant]
time_points = np.array([5,10,20,40,60,80,100,130,150,200]).astype(int)
number_parameters = len(true_parameter_values)

# Model definition
class GeneExpressionModel(gillespy2.Model):
    def __init__(self, parameter_values, initial_conditions, mode):
        super().__init__('GeneExpressionModel')
        for name, expression in parameter_values.items():
            self.add_parameter(gillespy2.Parameter(name=name, expression=expression))
        species_list = [
            gillespy2.Species(name='G_off', initial_value=initial_conditions.get('G_off'), mode=mode),
            gillespy2.Species(name='G_on', initial_value=initial_conditions.get('G_on'), mode=mode),
            gillespy2.Species(name='R_n', initial_value=initial_conditions.get('R_n'), mode=mode),
            gillespy2.Species(name='R_c', initial_value=initial_conditions.get('R_c'), mode=mode),
            gillespy2.Species(name='P', initial_value=initial_conditions.get('P'), mode=mode)
        ]
        self.add_species(species_list)
        reactions = [
            gillespy2.Reaction(name='gene_activation', reactants={species_list[0]: 1}, products={species_list[1]: 1}, rate=self.listOfParameters['k_on']),
            gillespy2.Reaction(name='gene_deactivation', reactants={species_list[1]: 1}, products={species_list[0]: 1}, rate=self.listOfParameters['k_off']),
            gillespy2.Reaction(name='mRNA_production', reactants={species_list[1]: 1}, products={species_list[1]: 1, species_list[2]: 1}, rate=self.listOfParameters['k_r']),
            gillespy2.Reaction(name='mRNA_transport', reactants={species_list[2]: 1}, products={species_list[3]: 1}, rate=self.listOfParameters['k_t']),
            gillespy2.Reaction(name='protein_production', reactants={species_list[3]: 1}, products={species_list[3]: 1, species_list[4]: 1}, rate=self.listOfParameters['k_p']),
            gillespy2.Reaction(name='nuclear_mRNA_decay', reactants={species_list[2]: 1}, products={}, rate=self.listOfParameters['gamma_rn']),
            gillespy2.Reaction(name='cytoplasm_mRNA_decay', reactants={species_list[3]: 1}, products={}, rate=self.listOfParameters['gamma_rc']),
            gillespy2.Reaction(name='protein_decay', reactants={species_list[4]: 1}, products={}, rate=self.listOfParameters['gamma_p'])
        ]
        self.add_reaction(reactions)
        self.timespan(np.linspace(0, 100, 101))

def initialize_model(parameter_values, initial_conditions, mode='continuous', apply_drug=False, inhibited_parameters=None):
    model = GeneExpressionModel(parameter_values, initial_conditions, mode=mode)
    for species in model.listOfSpecies.values():
        species.mode = mode
    if apply_drug and inhibited_parameters is not None:
        for param, value in inhibited_parameters.items():
            if param in model.listOfParameters:
                model.listOfParameters[param].expression = str(value)
    return model

def run_simulation_phase(model_initializer, parameter_values, initial_conditions, simulation_end, number_of_trajectories, apply_drug=False, inhibited_parameters=None, simulation_type='discrete', burn_in_time=None):
    model = model_initializer(parameter_values, initial_conditions, mode=simulation_type, apply_drug=apply_drug, inhibited_parameters=inhibited_parameters)
    total_timespan = np.linspace(0, simulation_end, num=int(simulation_end) + 1)
    model.timespan(total_timespan)
    if simulation_type == 'discrete':
        results = model.run(solver=gillespy2.TauLeapingSolver, number_of_trajectories=number_of_trajectories)
        list_all_results = []
        for n in range(number_of_trajectories):
            species_trajectories = {species: results[n][species] for species in model.listOfSpecies.keys()}
            if burn_in_time is not None:
                burn_in_index = burn_in_time if burn_in_time < len(total_timespan) else len(total_timespan) - 1
                species_trajectories = {species: species_trajectories[species][burn_in_index:] for species in model.listOfSpecies.keys()}
            list_all_results.append(species_trajectories)
        return list_all_results
    else:
        result = model.run(solver=gillespy2.ODESolver)
        if burn_in_time is not None:
            trajectories_species = {species: result[0][species][burn_in_time:] for species in model.listOfSpecies.keys()}
        else:
            trajectories_species = {species: result[0][species] for species in model.listOfSpecies.keys()}
        return trajectories_species

def simulate_model(parameter_values, initial_conditions, total_simulation_time, simulation_type='continuous', burn_in_time=None, drug_application_time=None, inhibited_parameters=None, number_of_trajectories=1):
    if burn_in_time is None or burn_in_time < 50:
        burn_in_time = None
    if burn_in_time is not None:
        end_time_initial_phase = drug_application_time + burn_in_time if drug_application_time is not None else total_simulation_time + burn_in_time
    else:
        end_time_initial_phase = drug_application_time if drug_application_time is not None else total_simulation_time
    trajectories_initial = run_simulation_phase(initialize_model, parameter_values=parameter_values, initial_conditions=initial_conditions, simulation_end=end_time_initial_phase, number_of_trajectories=number_of_trajectories, apply_drug=False, inhibited_parameters=None, simulation_type=simulation_type, burn_in_time=burn_in_time)
    
    if drug_application_time is not None:
        drug_simulation_end = total_simulation_time - drug_application_time
        updated_initial_conditions = {species: np.max((0,trajectories_initial[species][-1])) for species in trajectories_initial}
        trajectories_drug = run_simulation_phase(initialize_model, parameter_values=parameter_values, initial_conditions=updated_initial_conditions, simulation_end=drug_simulation_end, number_of_trajectories=1, apply_drug=True, inhibited_parameters=inhibited_parameters, simulation_type=simulation_type)
        trajectories_species = {species: np.concatenate([trajectories_initial[species], trajectories_drug[species][1:]]) for species in trajectories_initial}
    else:
        trajectories_species = trajectories_initial
    time = np.linspace(0, total_simulation_time, num=total_simulation_time + 1)
    return time, trajectories_species

# Define a proper prior that accounts for the parameter bounds
def bg_logprior(parameter):
    return -0.5 * np.sum((parameter - BG_MU)**2 / BG_SIGMA**2)

def Loglikelihood(parameters, observations_data, drug_application_time=0, total_simulation_time=201, time_points=None):
    parameter_values = {
        'k_on': parameters[0], 'k_off': parameters[1], 'k_r': parameters[2], 'k_t': parameters[3], 'k_p': parameters[4],
        'gamma_rn': parameters[5], 'gamma_rc': parameters[6], 'gamma_p': parameters[7], 'inhibition_constant': parameters[8]
    }
    initial_conditions = {'G_off': 0, 'G_on': 1, 'R_n': 0, 'R_c': 0, 'P': 0}
    inhibited_parameters = {'k_t': parameter_values['k_t'] * parameter_values['inhibition_constant']}
    time, concentrations_species =simulate_model(parameter_values, 
                                             initial_conditions, 
                                             total_simulation_time, 
                                             simulation_type='continuous', 
                                             burn_in_time=0, 
                                             drug_application_time=drug_application_time, 
                                             inhibited_parameters=inhibited_parameters, )
    try:
        time, concentrations_species =simulate_model(parameter_values, 
                                             initial_conditions, 
                                             total_simulation_time, 
                                             simulation_type='continuous', 
                                             burn_in_time=0, 
                                             drug_application_time=drug_application_time, 
                                             inhibited_parameters=inhibited_parameters, )
        if time_points is not None:
            y_R_n = concentrations_species['R_n'][time_points]
            y_R_c = concentrations_species['R_c'][time_points]
            y_P = concentrations_species['P'][time_points]
            SIGMA = 1
            loglikelihood = 0.0
            for i in range(len(observations_data[0])):  # Assuming observations[0] is correctly indexed
                loglikelihood += (np.sum(observations_data[0][i]) - y_P[i])**2 / (2 * SIGMA**2)
                loglikelihood += (np.sum(observations_data[1][i]) - y_R_n[i])**2 / (2 * SIGMA**2)
                loglikelihood += (np.sum(observations_data[2][i]) - y_R_c[i])**2 / (2 * SIGMA**2)
    except:
        loglikelihood = 1e20
        time = None
        concentrations_species = None
    return -loglikelihood, time, concentrations_species

def adaptive_metropolis(log_target_pdf, start, chain_len, initial_scale=0.1, rng=np.random.default_rng()):
    pdim = len(start)
    samples = np.zeros((chain_len, pdim))
    log_target_pdfs = np.zeros(chain_len)
    samples[0, :] = start
    log_target_pdfs[0] = log_target_pdf(samples[0, :])
    nacc = 0
    cov_matrix = initial_scale * np.eye(pdim)  # Start with a scaled identity matrix
    cov_matrix_adjustment = 1.0e-6 * (2.4 ** 2) / pdim
    update_freq = 1000  # Adjust the frequency of updates to covariance matrix
    # Modify tqdm initialization to use `miniters` and `mininterval`
    pbar = tqdm(total=chain_len - 1, miniters=1000, mininterval=1)
    update_counter = 0  # Initialize counter to manage tqdm updates
    for i in range(1, chain_len):
        in_range=False
        while in_range == False:
            xpropose = rng.multivariate_normal(samples[i-1, :], cov_matrix)
            transformed_sample = 10.0**xpropose
            if np.all(transformed_sample >= min_value_parameters) and np.all(transformed_sample <= max_value_parameters) and (transformed_sample[-1]<1):
                in_range=True
        logpipropose = log_target_pdf(xpropose)
        logu = np.log(rng.uniform())
        if logu < (logpipropose - log_target_pdfs[i-1]):
            samples[i, :] = xpropose
            log_target_pdfs[i] = logpipropose
            nacc += 1
        else:
            samples[i, :] = samples[i-1, :]
            log_target_pdfs[i] = log_target_pdfs[i-1]
        # Update tqdm progress bar and description only every 1000 iterations
        if i % 1000 == 0:
            #print(f"Debug: Iteration {i}, Likelihood = {logpipropose:.2f}, Acceptance Rate = {nacc / i:.4f}")
            #pbar.set_description(f"Iteration {i}")
            pbar.set_postfix(ll=f"{logpipropose:.2f}", ar=f"{nacc / i:.2f}")
            pbar.update(1000 - update_counter)  # Update the progress bar
            update_counter = 0  # Reset the update counter after the bar has been updated
        else:
            update_counter += 1  # Increment counter
        if i % update_freq == 0:
            new_cov_matrix = np.cov(samples[:i+1, :], rowvar=False) + cov_matrix_adjustment * np.eye(pdim)
            if np.all(np.linalg.eigvals(new_cov_matrix) > 0):
                cov_matrix = new_cov_matrix
    # Ensure the final state of the progress bar is full
    pbar.update(1000 - update_counter)
    pbar.close()
    return samples, log_target_pdfs

# Loading observable data
folder_simulated_data = current_dir.joinpath('simulated_data')
simulated_time_points = np.load(folder_simulated_data.joinpath('time_points_snapshots.npy'))
simulated_data_protein = np.load(folder_simulated_data.joinpath('snapshots_total_Protein_.npy'))
simulated_data_Rn = np.load(folder_simulated_data.joinpath('snapshots_RNA_nucleus.npy'))
simulated_data_Rc = np.load(folder_simulated_data.joinpath('snapshots_RNA_cytosol.npy'))

# Calculating mean values and saving in list with shape (n_species, n_time_points)
observations_data = [np.mean(simulated_data_protein, axis=0), np.mean(simulated_data_Rn, axis=0), np.mean(simulated_data_Rc, axis=0)]

# Initial values for the chain
# parameter_symbols = ['k_on', 'k_off', 'k_r', 'k_t', 'k_p', 'gamma_rn', 'gamma_rc', 'gamma_p', 'inhibition_constant']
BG_MU = np.array([-1.0, -1.0, 1.0, -2.0, -1.0, -1.0, -1.0, -1.0, -1.0])
BG_SIGMA = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
def bg_prior_sample(n_sample: int, rng=np.random.default_rng()) -> np.ndarray:
    while True:
        prior_sample = rng.normal(loc=BG_MU, scale=BG_SIGMA, size=(n_sample, BG_MU.shape[0]))
        transformed_sample = 10.0**prior_sample[0,:]
        if np.all(transformed_sample >= min_value_parameters) and np.all(transformed_sample <= max_value_parameters) and (transformed_sample[-1]<1):
            return prior_sample
chain_start =  bg_prior_sample(1)[0]
print('Initial guess:', 10.0**chain_start)

# Assuming you have a function to calculate log likelihood and log prior
def negative_log_posterior(theta_untransformed, observations_data, drug_application_time, total_simulation_time, time_points):
    theta = 10.0 ** theta_untransformed  # Apply the transformation within the function
    log_prior = bg_logprior(theta_untransformed)  # Assuming you have a function for the log prior
    log_likelihood = Loglikelihood(theta, observations_data, drug_application_time, total_simulation_time, time_points)[0]
    return (log_prior + log_likelihood)  # Negative because minimize seeks to minimize the function
# running the MCMC
if run_ms:
    bg_log_target_partial = partial(negative_log_posterior, observations_data=observations_data, drug_application_time=drug_application_time, total_simulation_time=total_simulation_time, time_points=time_points)
    #bg_chain, bg_logpos = adaptive_metropolis(bg_log_target_partial, start=chain_start, chain_len=chain_length, cov_matrix=np.eye(len(BG_MU)))
    bg_chain, bg_logpos = adaptive_metropolis(bg_log_target_partial, start=chain_start, chain_len=chain_length, initial_scale=0.1, rng=np.random.default_rng())
    np.save(folder_outputs.joinpath('bg_chain.npy'), bg_chain)
    np.save(folder_outputs.joinpath('bg_logpos.npy'), bg_logpos)
else:
    bg_chain = np.load(folder_outputs.joinpath('bg_chain.npy'))
    bg_logpos = np.load(folder_outputs.joinpath('bg_logpos.npy'))
# plot the ODE model with the best fit parameters
best_fit = np.mean(bg_chain[burnin_time:], axis=0)
best_fit = np.round(10.0**( best_fit),3)
ll, time, concentrations_species = Loglikelihood(best_fit, observations_data=observations_data,drug_application_time=drug_application_time,total_simulation_time=total_simulation_time, time_points=time_points)
print(ll)
# solving the model with the true parameters
print('true parameters:', true_parameter_values)
ll, time, concentrations_species_true = Loglikelihood(true_parameter_values, observations_data=observations_data,drug_application_time=drug_application_time,total_simulation_time=total_simulation_time, time_points=time_points)
print(ll)

# Plotting the ODE model and the experimental data only for variables P and R_n
plt.figure(figsize=(8, 5))
# plotting for observations_data
plt.plot(time_points, observations_data[0], 'o', color=species_colors['P'], label=' Data (P)', markersize=8)
plt.plot(time_points, observations_data[1], 'o', color=species_colors['R_n'], label=' Data (R_n)', markersize=8)
plt.plot(time_points, observations_data[2], 'o', color=species_colors['R_c'], label=' Data (R_c)', markersize=8)
# plotting model fit
plt.plot(time, concentrations_species['P'],   color=species_colors['P'], label='Model Fit (P)', lw=4)
plt.plot(time, concentrations_species['R_n'], color=species_colors['R_n'], label='Model Fit (R_n)', lw=4)
plt.plot(time, concentrations_species['R_c'], color=species_colors['R_c'], label='Model Fit (R_c)', lw=4)
# plotting for concentrations_species_true
plt.plot(time, concentrations_species_true['P'], color=species_colors['P'], label='True Model (P)', lw=2, linestyle='--')
plt.plot(time, concentrations_species_true['R_n'], color=species_colors['R_n'], label='True Model (R_n)', lw=2, linestyle='--')
plt.plot(time, concentrations_species_true['R_c'], color=species_colors['R_c'], label='True Model (R_c)', lw=2, linestyle='--')
if drug_application_time > 0:
    plt.axvline(x=drug_application_time, color='k', linestyle='--', label='Drug Application')
# Adding labels and legend
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('ODE Model vs Data')
plt.legend(bbox_to_anchor=(1.3, 1),loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
#plt.ylim(0, 55)
# save the figure
plt.savefig(folder_outputs.joinpath('model_fit.png'), dpi=400, bbox_inches="tight")
plt.show()

fig, axs = plt.subplots(1, number_parameters, figsize=(20, 4))  # Adjusted for a horizontal layout
fig.set_tight_layout(True)
# Setting titles for the top row for "MCMC Chain" and the bottom row for "Autocorrelation"
for i in range(number_parameters):
    #if i == 3:
    axs[i].set_title(parameter_symbols[i], fontsize=12)
        #axs[1, i].set_title("Autocorrelation", fontsize=20)
    # MCMC chain plot
    axs[i].plot(10.0**(bg_chain[burnin_time:, i]), color='lightslategray', lw=2)
    axs[i].axhline(true_parameter_values[i], color='orangered', lw=3)
    #axs[i].set_ylabel(parameter_symbols[i])
    #axs[i].set_ylabel(parameter_symbols[i])
    #axs[i].set_xlabel('MH-steps')
    #axs[i].set_xlabel('MH-steps')
    # Autocorrelation plot
    # Note: Adjust the autocorrelation calculation based on your version of emcee
    #autocorr = emcee.autocorr.function_1d(bg_chain[burnin_time:, i]) if hasattr(emcee.autocorr, 'function_1d') else np.correlate(bg_chain[burnin_time:, i], bg_chain[burnin_time:, i], mode='full')[len(bg_chain[burnin_time:, i])-1:] / len(bg_chain[burnin_time:, i])
    #axs[1, i].plot(autocorr, color='lightslategray', lw=3)
fig.savefig(folder_outputs.joinpath("mcmc_trajectories_horizontal.png"), dpi=400, bbox_inches="tight")