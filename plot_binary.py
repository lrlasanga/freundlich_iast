import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# Define the extended Freundlich isotherm equation in its logarithmic form
def freundlich_extended_log(Ce_B, KF_B, n_B, alpha_B, Ce_A):
    return np.log(KF_B) + (1 / n_B) * (np.log(Ce_B) - np.log(1 + alpha_B * Ce_A))

def fit_func(Ce_B, KF_B, n_B, alpha_B):
    return freundlich_extended_log(Ce_B, KF_B, n_B, alpha_B, Ce_A)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and draw adsorption graph')
    parser.add_argument('--file', type=str, required=True, help='Path to the csv file')
    parser.add_argument('--A', type=str, required=True, help='Name of the component A')
    parser.add_argument('--B', type=str, required=True, help='Name of the component B')
    args = parser.parse_args()
    df_set = pd.read_csv(args.file)
    Ce_A = df_set["CA"].to_numpy()
    Ce_B = df_set["CB"].to_numpy()
    qe_B = df_set["qB"].to_numpy()
    qe_A = df_set["qA"].to_numpy()
    log_qe_B = np.log(qe_B)
    log_qe_A = np.log(qe_A)
    q_A_Values = {}
    q_B_Values = {}

    log_Ce_B = np.log(Ce_B)
    log_Ce_A = np.log(Ce_A)
    KB = 0
    NB = 0
    NA = 0
    KA = 0
    for index, row in df_set.iterrows():
        KB = row["KB"]
        NB = row["nB"]
        KA = row["KA"]
        NA = row["nA"]
        break

    initial_guess = [KB, NB, 0.1]
    popt, pcov = curve_fit(fit_func, Ce_B, log_qe_B, p0=initial_guess)
    KF_B_opt, n_B_opt, alpha_B_opt = popt
    initial_guess = [KA, NA, 0.1]
    popt, pcov = curve_fit(fit_func, Ce_A, log_qe_A, p0=initial_guess)
    KF_A_opt, n_A_opt, alpha_A_opt = popt

    for CA in Ce_A:
        qe_B_values = KB * ((Ce_B) / (1 + alpha_B_opt * CA)) ** (1 / NB)
        q_A_Values[CA] = qe_B_values

    for CB in Ce_B:
        qe_B_values = KA * ((Ce_A) / (1 + alpha_A_opt * CB)) ** (1 / NA)
        q_B_Values[CB] = qe_B_values

    for CA in Ce_A:
        qe_B = q_A_Values[CA]
        log_qe_B = np.log(qe_B)
        plt.scatter(log_Ce_B, log_qe_B)
        plt.plot(log_Ce_B, log_qe_B, linestyle='-', label=f'{args.A} = {CA}')
    plt.xlabel('log(Ce)')
    plt.ylabel('log(qe)')
    plt.title(f'{args.B} absorption against {args.A}')
    plt.grid(True)
    plt.legend()
    plt.show()

    for CB in Ce_B:
        qe_A = q_B_Values[CB]
        log_qe_A = np.log(qe_A)
        plt.scatter(log_Ce_A, log_qe_A)
        plt.plot(log_Ce_A, log_qe_A, linestyle='-', label=f'{args.B} = {CB}')
    plt.xlabel('log(Ce)')
    plt.ylabel('log(qe)')
    plt.title(f'{args.A} absorption against {args.B}')
    plt.grid(True)
    plt.legend()
    plt.show()
