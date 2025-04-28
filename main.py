import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from neural_network import NeuralNetworkModel  # Assuming you have a neural network model defined in a separate file

# read the data from the excel file
data = pd.read_excel('PMK Karanganyar.xlsx')

# Ensure the 'date' column is in datetime format and sort by date
date = data['tanggal_pelaporan_p']

# Ensure the 'date' column is in datetime format and remove duplicates
data['tanggal_pelaporan_p'] = pd.to_datetime(data['tanggal_pelaporan_p'])
data = data.groupby('tanggal_pelaporan_p', as_index=False).agg({
    'total_sakit': 'sum',
    'total_sembuh': 'sum'
}).reset_index(drop=True)

# change the name of the columns
infectious = data['total_sakit']
recovered = data['total_sembuh']

#fill the missing values with 0
data = data.fillna(0)

# save the data to a new excel file
data.to_excel('PMK Karanganyar.xlsx', index=False)

# Extract initial conditions from the data
I0 = infectious.iloc[0]  # Initial number of infectious individuals
R0 = recovered.iloc[0]  # Initial number of recovered individuals
E0 = 6                         # Assume no exposed individuals initially
N = 100                      # Total population (adjust as needed)
S0 = N - I0 - R0 - E0         # Initial number of susceptible individuals

# Parameters for the SEIR model
beta = 0.6   # Infection rate
sigma = 1/2  # Incubation rate (1/average incubation period)
gamma = 1/5  # Recovery rate (1/average infectious period)

# Time points (days)
t = np.linspace(0, len(data) - 1, len(data))

# SEIR model differential equations 
def deriv(t, y, N, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# initial conditions vector
y0 = S0, E0, I0, R0 

# solve the SEIR model
sol = solve_ivp(deriv, [0, len(data) - 1], y0, args=(N, beta, sigma, gamma), t_eval=t)

# Extract the results
S, E, I, R = sol.y

# Reproductive number (R0) calculation
R0 = beta / gamma

# # Calculate residuals
# infectious_residuals = infectious - I
# recovered_residuals = recovered - R

# # Train neural networks for residuals
# print("Training neural network for infectious residuals...")
# nn_infectious = NeuralNetworkModel(hidden_layer_sizes=(64, 64), max_iter=500)
# X = np.arange(len(data)).reshape(-1, 1)  # Use time as the feature
# nn_infectious.train(X, infectious_residuals)

# print("Training neural network for recovered residuals...")
# nn_recovered = NeuralNetworkModel(hidden_layer_sizes=(64, 64), max_iter=500)
# nn_recovered.train(X, recovered_residuals)

# # Predict residuals using the neural networks
# infectious_residuals_pred = nn_infectious.predict(X)
# recovered_residuals_pred = nn_recovered.predict(X)

# final_infectious_pred = I + infectious_residuals_pred
# final_recovered_pred = R + recovered_residuals_pred

# save results as npy
np.save('final_infectious_pred.npy', I)
np.save('final_recovered_pred.npy', R)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, E, 'y', label='Exposed')
plt.plot(t, I, 'r', label='Infectious')
plt.plot(t, R, 'g', label='Recovered')
plt.scatter(range(len(data)), infectious, color='red', label='Actual Infectious Data', alpha=0.6)
plt.scatter(range(len(data)), recovered, color='green', label='Actual Recovered Data', alpha=0.6)
# plt.plot(t, final_infectious_pred, 'm', label='Infectious')
# plt.plot(t, final_recovered_pred, 'c', label='Recovered')
plt.title(f"SEIR Model Dynamics (R0 = {R0:.2f})")
plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.grid()
plt.show()
