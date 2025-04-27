import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# read the data from the excel file
data = pd.read_excel('PMK Karanganyar.xlsx')

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
E0 = 0                         # Assume no exposed individuals initially
N = 100                      # Total population (adjust as needed)
S0 = N - I0 - R0 - E0         # Initial number of susceptible individuals

# Parameters for the SEIR model
beta = 0.44   # Infection rate
sigma = 1/14  # Incubation rate (1/average incubation period)
gamma = 1/8  # Recovery rate (1/average infectious period)

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

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, E, 'y', label='Exposed')
plt.plot(t, I, 'r', label='Infectious')
plt.plot(t, R, 'g', label='Recovered')
plt.scatter(range(len(data)), infectious, color='red', label='Actual Infectious Data', alpha=0.6)
plt.scatter(range(len(data)), recovered, color='green', label='Actual Recovered Data', alpha=0.6)
plt.title(f"SEIR Model Dynamics (R0 = {R0:.2f})")
plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.grid()
plt.show()
