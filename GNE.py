# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 22:39:51 2023
Edit on Fri Mar 13 2026
@author: Haoran Liu

Ref: Algorithm 1 in 
"Approaching Prosumer Social Optimum via Energy Sharing With Proof of Convergence"
"""

import numpy as np
from docplex.mp.model import Model
import matplotlib.pyplot as plt

# ── Global Parameters ──────────────────────────────────────────────────────
a = 100
homes = 3  

alpha1 = [0.015, 0.008, 0.011]
alpha2 = [0.038, 0.047, 0.056]
beta1  = [-0.008, -0.014, -0.009]  
beta2  = [0.8, 0.5, 0.4]

P_BOUNDS = [(0, 20), (0, 25), (0, 30)]
D_BOUNDS = [(5, 15), (7, 18), (10, 25)]

def lower_level(price_val, home_idx):
    L_model = Model(log_output=False)
    p_lb, p_ub = P_BOUNDS[home_idx]
    D_lb, D_ub = D_BOUNDS[home_idx]

    p_var = L_model.continuous_var(lb=p_lb, ub=p_ub, name='p')
    D_var = L_model.continuous_var(lb=D_lb, ub=D_ub, name='D')

    # Objective
    cost = (
        alpha1[home_idx] * p_var**2 + alpha2[home_idx] * p_var      
        - (beta1[home_idx] * D_var**2 + beta2[home_idx] * D_var)    
        + (D_var - p_var)**2 / (2 * a * (homes - 1))                
        + price_val * (D_var - p_var)                               
    )
    
    L_model.minimize(cost)
    sol = L_model.solve()
    return (sol[p_var], sol[D_var]) if sol else (None, None)

def main():
    max_iterations = 100
    tolerance = 1e-6 

    # ── Initialize NumPy Arrays ──────────────────────────────────────────────
    price = np.zeros(max_iterations + 1)
    p = np.zeros((homes, max_iterations + 1))
    D = np.zeros((homes, max_iterations + 1))
    b = np.zeros((homes, max_iterations + 1))

    print("=" * 75)
    print(f"{'Iter':>4} | {'Price[k]':>15} | {'Price[k+1]':>15} | {'Delta':>12}")
    print("-" * 75)

    final_k = 0
    for k in range(max_iterations):
        # 1. Prosumer Update (Lower Level)
        for i in range(homes):
            p[i, k+1], D[i, k+1] = lower_level(price[k], i)
            if np.isnan(p[i, k+1]):
                print(f"\n[!] Solver failed at Iteration {k} for Prosumer {i}.")
                return
            b[i, k+1] = (D[i, k+1] - p[i, k+1]) + a * price[k]

        # 2. Platform Update 
        price[k+1] = np.sum(b[:, k+1]) / (a * homes)

        # 3. Convergence Check
        price_change = abs(price[k+1] - price[k])
        print(f"{k:4d} | {price[k]:15.8f} | {price[k+1]:15.8f} | {price_change:12.2e}")

        final_k = k
        if price_change < tolerance:
            print(f"\n[OK] Algorithm converged at iteration {k}.")
            break
    else:
        print("\n[!] Maximum iterations reached.")

    # ── Final Results ───────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"Final Equilibrium Price : {price[final_k+1]:.6f}")
    print("-" * 75)
    
    for i in range(homes):
        net_energy = D[i, final_k+1] - p[i, final_k+1]
        status = "Buyer " if net_energy >= 0 else "Seller"
        print(f" Prosumer {i+1}: Gen={p[i, final_k+1]:7.4f}, Dem={D[i, final_k+1]:7.4f}, "
              f"Net={net_energy:+8.4f} ({status})")
    
    # Verify Market Clearing: Sum of net demands should be zero
    clearing_residual = np.sum(D[:, final_k+1] - p[:, final_k+1])
    print(f" Market Clearing Residual: {clearing_residual:.2e}")
    print("=" * 75)

    # ── Visualization ───────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(price[:final_k+2], color='blue', marker='o', markersize=4)
    plt.title("Price Convergence Process (Algorithm 1)")
    plt.xlabel("Iteration (k)")
    plt.ylabel(r"Price ($\lambda$)")
    plt.show()

if __name__ == "__main__":
    main()