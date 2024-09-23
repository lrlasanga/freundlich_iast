import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def freundlich_extended_log(Ce_tuple, K, n, alpha, beta):
    Ce_B, Ce_A, Ce_C = Ce_tuple
    return np.log(K) + n * (np.log(1 + alpha*Ce_A + beta*Ce_C) - np.log(Ce_B))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and draw adsorption graph')
    parser.add_argument('--file', type=str, required=True, help='Path to the csv file')
    parser.add_argument('--A', type=str, required=True, help='Name of the component A')
    parser.add_argument('--B', type=str, required=True, help='Name of the component B')
    parser.add_argument('--C', type=str, required=True, help='Name of the component C')
    args = parser.parse_args()
    df_set = pd.read_csv(args.file)
    Ce_A = df_set["CA"].to_numpy()
    Ce_B = df_set["CB"].to_numpy()
    Ce_C = df_set["Cc"].to_numpy()
    qe_B = df_set["qB"].to_numpy()
    qe_A = df_set["qA"].to_numpy()
    qe_C = df_set["qc"].to_numpy()
    log_qe_B = np.log(qe_B)
    log_qe_A = np.log(qe_A)
    log_qe_C = np.log(qe_C)
    q_A_Values = {}
    q_B_Values = {}
    q_C_Values = {}

    log_Ce_B = np.log(Ce_B)
    log_Ce_A = np.log(Ce_A)
    log_Ce_C = np.log(Ce_C)
    KB = 0
    NB = 0
    NA = 0
    KA = 0
    KC = 0
    NC = 0
    for index, row in df_set.iterrows():
        KB = row["KB"]
        NB = row["nB"]
        KA = row["KA"]
        NA = row["nA"]
        KC = row["Kc"]
        NC = row["nc"]
        break

    initial_guess = [KB, NB, 0.1, 0.1]
    popt, pcov = curve_fit(freundlich_extended_log, (Ce_B, Ce_A, Ce_C), log_qe_B, p0=initial_guess)
    KF_B_opt, n_B_opt, alpha_B_opt, beta_B_opt = popt
    initial_guess = [KA, NA, 0.1, 0.1]
    popt, pcov = curve_fit(freundlich_extended_log, (Ce_B, Ce_A, Ce_C), log_qe_A, p0=initial_guess)
    KF_A_opt, n_A_opt, alpha_A_opt, beta_A_opt = popt
    initial_guess = [KC, NC, 0.1, 0.1]
    popt, pcov = curve_fit(freundlich_extended_log, (Ce_B, Ce_A, Ce_C), log_qe_C, p0=initial_guess)
    KF_C_opt, n_C_opt, alpha_C_opt, beta_C_opt = popt

    for idx, CA in enumerate(Ce_A):
        qe_A_values = KB * ((1 + (alpha_B_opt * CA) + (beta_B_opt * Ce_C[idx]))/Ce_B) ** NB
        q_A_Values[CA] = qe_A_values

    for idx, CB in enumerate(Ce_B):
        qe_B_values = KA * ((1 + (alpha_A_opt * CB) + (beta_A_opt * Ce_C[idx]))/Ce_A) ** NA
        q_B_Values[CB] = qe_B_values

    for idx, CC in enumerate(Ce_C):
        qe_C_values = KC * ((1 + (alpha_C_opt * CC) + (beta_C_opt * Ce_A[idx]))/Ce_B) ** NC
        q_C_Values[CC] = qe_C_values

    for idx, CA in enumerate(Ce_A):
        qe_B = q_A_Values[CA]
        log_qe_B = np.log(qe_B)
        plt.scatter(log_Ce_B, log_qe_B)
        plt.plot(log_Ce_B, log_qe_B, linestyle='-', label=f'{args.B} = {CA} {args.C} = {Ce_C[idx]}')
    plt.xlabel('log(Ce)')
    plt.ylabel('log(qe)')
    plt.title(f'{args.A} absorption against {args.B} and {args.C}')
    plt.grid(True)
    plt.legend()
    plt.show()

    for idx, CB in enumerate(Ce_B):
        qe_A = q_B_Values[CB]
        log_qe_A = np.log(qe_A)
        plt.scatter(log_Ce_A, log_qe_A)
        plt.plot(log_Ce_A, log_qe_A, linestyle='-', label=f'{args.A} = {CB} {args.C} = {Ce_C[idx]}')
    plt.xlabel('log(Ce)')
    plt.ylabel('log(qe)')
    plt.title(f'{args.B} absorption against {args.A} and {args.C}')
    plt.grid(True)
    plt.legend()
    plt.show()

    for idx, CC in enumerate(Ce_C):
        qe_C = q_C_Values[CC]
        log_qe_C = np.log(qe_C)
        plt.scatter(log_Ce_C, log_qe_C)
        plt.plot(log_Ce_C, log_qe_C, linestyle='-', label=f'{args.B} = {CC} {args.A} = {Ce_A[idx]}')
    plt.xlabel('log(Ce)')
    plt.ylabel('log(qe)')
    plt.title(f'{args.C} absorption against {args.A} and {args.B}')
    plt.grid(True)
    plt.legend()
    plt.show()

