import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import math

H = 50 
K = 12   

row1 = [0.4638, 0.0379, 0.0666, 0.0330, 0.0713, 0.0755,
        0.0508, 0.0621, 0.0078, 0.0013, 0.0610, 0.0689]
row2 = [0.3917, 0.0243, 0.0626, 0.0578, 0.0591, 0.0859,
        0.0592, 0.0656, 0.0656, 0.0024, 0.0541, 0.0717]
alpha = np.array(
    [row1] * 10 +  
    [row2] * 40     
)

c = np.array([0.6]*50)  
I = np.array([
    91173, 6535, 24245, 19371,  98578, 16843, 32185, 10735, 13839, 44073,
    45928, 68321, 33664, 44114, 34749, 33920, 67955, 56167, 29431, 72030,
    53354, 72978, 33275, 41815, 29244, 40216, 23209, 14544, 128840, 38324,
    60404, 40054, 46445, 31707, 65700, 120789, 17869, 37317, 68442, 32050,
    63998, 44216, 42230, 14198, 64530, 86935, 15854, 109796, 13751, 56353])  

beta = np.array([1/x for x in I])


R = 250000 
epsilon = 1e-8  


s = cp.Variable((H, K), nonneg=True) 
tau = cp.Variable(H, nonneg=True)


objective_terms = []
for h in range(H):
    log_terms = cp.log(s[h, :])
    weighted_logs = cp.multiply(alpha[h, :], log_terms)
    objective_terms.append(beta[h] * cp.sum(weighted_logs))
objective = cp.sum(objective_terms)


constraints = []

for h in range(H):
    constraints.append(cp.sum(s[h, :]) == c[h] * (1 - tau[h]) * I[h])

constraints.append(cp.sum(tau @ I) == R)

constraints.append(tau <= 1 - epsilon)


prob = cp.Problem(cp.Maximize(objective), constraints)
prob.solve()


print("Status:", prob.status)
if prob.status == 'optimal':
    print("Optimal Objective Value:", prob.value)
    print("Optimal s values:")
    print(s.value)
    print("Optimal tau values:")
    print(tau.value)
else:
    print("No optimal solution found. Status:", prob.status)

if prob.status == 'optimal':
   
    plt.style.use('ggplot')

    sorted_indices = np.argsort(I)
    I_sorted = I[sorted_indices]
    tauI_sorted = tau.value[sorted_indices] * I[sorted_indices]
    group_labels = [f"Group {i+1}" for i in sorted_indices]


    fig, ax = plt.subplots(figsize=(12, 8))
    

    x = np.arange(H)
    width = 0.35
    

    bars1 = ax.barh(x - width/2, I_sorted, width, label='Income $I^h$', color='#1f77b4')
    bars2 = ax.barh(x + width/2, tauI_sorted, width, label='Tax Paid $\\tau^h I^h$', color='#ff7f0e')
    

    ax.axvline(R, color='red', linestyle='--', linewidth=2, label=f'Required Revenue $R={R}$')
    

    ax.set_yticks(x)
    ax.set_yticklabels(group_labels)
    ax.set_xlabel('Amount')
    ax.set_title('Income vs. Tax Paid Across Groups (Sorted by Income)')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.hist(tau.value, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Tax Rate (τ^h)')
    plt.ylabel('Number of Groups')
    plt.title('Distribution of Optimal Tax Rates')
    plt.show()

else:
    print("No solution to plot.")
