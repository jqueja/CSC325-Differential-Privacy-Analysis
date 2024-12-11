"""
 Important to note that python's float type has the same precision as the C++
 double.
"""

import sys  # isort:skip

sys.path.append("../pydp")  # isort:skip

# stdlib
import os
from pathlib import Path

# third party
import pandas as pd
import importlib
random = importlib.import_module('random')

# pydp absolute
import pydp as dp
from pydp.algorithms.laplacian import BoundedMean

'''
Epsilon (ε) is the most important parameter in differential privacy. 
It controls how much "privacy loss" is allowed when releasing information about a dataset.
Lower is more private

Sensitivity (Δ) measures the maximum change in the output (like the sum, mean, or count) 
that occurs if you change just one data point in the dataset.
'''
EPSILON = 10
SENSITIVITY = 1

from houseavg import ClassReporter

# Creating a class ClassReporter
class ClassReporter:

    # Function to read the csv file and creating a dataframe
    def __init__(self, data_filename, epsilon):
        self.data_filename = data_filename
        self.epsilon = epsilon
        self._epsilon = epsilon
        self._privacy_budget = float(1.0)

        self._df = pd.read_csv(data_filename, 
                        dtype={
                            'id': 'str',
                            'student_name': 'str',
                            'house': 'str',
                            'potions': 'float',
                            'herbology': 'float',
                            'darkarts': 'float',
                            'flying': 'float'
                        })

    # Function to return total number of in dataset.
    def sum_hours(self) -> int:
        return len(self._df)
    
    # Function to return average grade for Gryffindor
    def mean_gryffindor_grades(self) -> float:
        gryffindor_df = self._df[self._df['house'] == 'Gryffindor']
        grade_columns = ['potions', 'herbology', 'darkarts', 'flying']
        total_grades = gryffindor_df[grade_columns].to_numpy().flatten()
        return total_grades.mean()
    
    # Function to return average grade for Slytherin
    def mean_slytherin_grades(self) -> float:
        gryffindor_df = self._df[self._df['house'] == 'Slytherin']
        grade_columns = ['potions', 'herbology', 'darkarts', 'flying']
        total_grades = gryffindor_df[grade_columns].to_numpy().flatten()
        return total_grades.mean()
    
    # Function to return average grade for Ravenclaw
    def mean_ravenclaw_grades(self) -> float:
        gryffindor_df = self._df[self._df['house'] == 'Ravenclaw']
        grade_columns = ['potions', 'herbology', 'darkarts', 'flying']
        total_grades = gryffindor_df[grade_columns].to_numpy().flatten()
        return total_grades.mean()
    
    # Function to return average grade for Hufflepuff
    def mean_hufflepuff_grades(self) -> float:
        gryffindor_df = self._df[self._df['house'] == 'Hufflepuff']
        grade_columns = ['potions', 'herbology', 'darkarts', 'flying']
        total_grades = gryffindor_df[grade_columns].to_numpy().flatten()
        return total_grades.mean()
    
    # Adds an entry to the Laplace mechanism and returns the noisy result.
    def add_noisy_entry(self, x) -> float:

        # Create a new mechanism for each entry
        laplace_mechanism = BoundedMean(
            epsilon=EPSILON, 
            l0_sensitivity=SENSITIVITY, 
            lower_bound=0,  # Grades are bounded 0 to 100
            upper_bound=100,
            dtype="float"
        )
        laplace_mechanism.add_entry(x)  # Add the student's grade to the Laplace mechanism
        return laplace_mechanism.result()  # Get the noisy result
    
    # Function to add noise to the students' grades before calculating the house average
    def mean_with_noise_local(self, house: str) -> float:

        # Filter for the house
        house_df = self._df[self._df['house'] == house]
        grade_columns = ['potions', 'herbology', 'darkarts', 'flying']
        
        # Add Laplace noise to each student's grades
        noisy_grades = house_df[grade_columns].map(
            lambda x: self.add_noisy_entry(x)
        )

        # Calculate the mean of all noisy grades
        total_noisy_grades = noisy_grades.values.flatten()
        
        # Return the mean of noisy grades
        return total_noisy_grades.mean()
    
    # Function to compute the house mean and then add Laplace noise (Global DP)
    def mean_with_noise_global(self, house: str) -> float:
        # Filter for the house
        house_df = self._df[self._df['house'] == house]
        grade_columns = ['potions', 'herbology', 'darkarts', 'flying']
        
        # Calculate the mean of all grades for this house without using numpy
        all_grades = house_df[grade_columns].values.flatten().tolist()  # Flatten and convert to a list
        true_mean = sum(all_grades) / len(all_grades)  # Calculate mean using sum and len
        
        # Create a Laplace mechanism to add noise to the house mean
        laplace_mechanism = BoundedMean(
            epsilon=EPSILON, 
            l0_sensitivity=SENSITIVITY, 
            lower_bound=0,  # Grades are bounded between 0 and 100
            upper_bound=100, 
            dtype="float"  # Allow float inputs
        )

        # Add the mean of all grades to the Laplace mechanism
        laplace_mechanism.add_entry(true_mean)  # Add the calculated house mean
        noisy_mean = laplace_mechanism.result()  # Get the noisy mean
        
        return noisy_mean

# get absolute path
path = Path(os.path.dirname(os.path.abspath(__file__)))

reporter = ClassReporter(path / "hogwarts_student_performance.csv", EPSILON)

print("Total Students:\t" + str(reporter.sum_hours()))

# Print the house grade average with no DP (True Data)
print("📊 True Data (No Differential Privacy Applied):")
true_gryffindor_avg = reporter.mean_gryffindor_grades()
true_ravenclaw_avg = reporter.mean_ravenclaw_grades()
true_slytherin_avg = reporter.mean_slytherin_grades()
true_hufflepuff_avg = reporter.mean_hufflepuff_grades()

print(f"🏠 Ravenclaw Avg:\t{true_ravenclaw_avg:.2f}")
print(f"🏠 Slytherin Avg:\t{true_slytherin_avg:.2f}")
print(f"🏠 Gryffindor Avg:\t{true_gryffindor_avg:.2f}")
print(f"🏠 Hufflepuff Avg:\t{true_hufflepuff_avg:.2f}")
print("\n")  # Add space for clarity


# Print the house grade averages with noise (Local Differential Privacy Applied)
print(f"📢 Local Differential Privacy (ε = {EPSILON}):")
local_gryffindor_noisy_mean = reporter.mean_with_noise_local('Gryffindor')
local_ravenclaw_noisy_mean = reporter.mean_with_noise_local('Ravenclaw')
local_slytherin_noisy_mean = reporter.mean_with_noise_local('Slytherin')
local_hufflepuff_noisy_mean = reporter.mean_with_noise_local('Hufflepuff')

print(f"🏠 Ravenclaw Noisy Avg:\t{local_ravenclaw_noisy_mean:.2f}")
print(f"🏠 Slytherin Noisy Avg:\t{local_slytherin_noisy_mean:.2f}")
print(f"🏠 Gryffindor Noisy Avg:\t{local_gryffindor_noisy_mean:.2f}")
print(f"🏠 Hufflepuff Noisy Avg:\t{local_hufflepuff_noisy_mean:.2f}")
print("\n")  # Add space for clarity


# Print the house grade averages with noise (Global Differential Privacy Applied)
print(f"📢 Global Differential Privacy (ε = {EPSILON}):")
global_gryffindor_noisy_mean = reporter.mean_with_noise_global('Gryffindor')
global_ravenclaw_noisy_mean = reporter.mean_with_noise_global('Ravenclaw')
global_slytherin_noisy_mean = reporter.mean_with_noise_global('Slytherin')
global_hufflepuff_noisy_mean = reporter.mean_with_noise_global('Hufflepuff')

print(f"🏠 Ravenclaw Global Noisy Avg:\t{global_ravenclaw_noisy_mean:.2f}")
print(f"🏠 Slytherin Global Noisy Avg:\t{global_slytherin_noisy_mean:.2f}")
print(f"🏠 Gryffindor Global Noisy Avg:\t{global_gryffindor_noisy_mean:.2f}")
print(f"🏠 Hufflepuff Global Noisy Avg:\t{global_hufflepuff_noisy_mean:.2f}")
print("\n")  # Add space for clarity


# Print final summary for comparison
print("📋 Summary of Averages")
print("───────────────────────────────────────────")
print(f"🏠 Ravenclaw True Avg: \t{true_ravenclaw_avg:.2f}")
print(f"🏠 Ravenclaw Local DP Avg: \t{local_ravenclaw_noisy_mean:.2f}")
print(f"🏠 Ravenclaw Global DP Avg: \t{global_ravenclaw_noisy_mean:.2f}")
print("\n")
print(f"🏠 Slytherin True Avg: \t{true_slytherin_avg:.2f}")
print(f"🏠 Slytherin Local DP Avg: \t{local_slytherin_noisy_mean:.2f}")
print(f"🏠 Slytherin Global DP Avg: \t{global_slytherin_noisy_mean:.2f}")
print("\n")
print(f"🏠 Gryffindor True Avg: \t{true_gryffindor_avg:.2f}")
print(f"🏠 Gryffindor Local DP Avg: \t{local_gryffindor_noisy_mean:.2f}")
print(f"🏠 Gryffindor Global DP Avg: \t{global_gryffindor_noisy_mean:.2f}")
print("\n")
print(f"🏠 Hufflepuff True Avg: \t{true_hufflepuff_avg:.2f}")
print(f"🏠 Hufflepuff Local DP Avg: \t{local_hufflepuff_noisy_mean:.2f}")
print(f"🏠 Hufflepuff Global DP Avg: \t{global_hufflepuff_noisy_mean:.2f}")
print("\n")  # Add space for clarity


import matplotlib.pyplot as plt
import numpy as np

# Plot true vs noisy averages
houses = ['Gryffindor', 'Ravenclaw', 'Slytherin', 'Hufflepuff']
true_means = [true_gryffindor_avg, true_ravenclaw_avg, true_slytherin_avg, true_hufflepuff_avg]
local_noisy_means = [local_gryffindor_noisy_mean, local_ravenclaw_noisy_mean, local_slytherin_noisy_mean, local_hufflepuff_noisy_mean]
global_noisy_means = [global_gryffindor_noisy_mean, global_ravenclaw_noisy_mean, global_slytherin_noisy_mean, global_hufflepuff_noisy_mean]

# Bar plot for comparison
x = np.arange(len(houses))
width = 0.2

plt.bar(x - width, true_means, width, label='True Avg', color='blue')
plt.bar(x, local_noisy_means, width, label='Local DP Avg', color='orange')
plt.bar(x + width, global_noisy_means, width, label='Global DP Avg', color='green')

plt.xlabel('Houses')
plt.ylabel('Average Grades')
plt.title('Comparison of True and Noisy Averages')
plt.xticks(x, houses)
plt.legend()
plt.show()


import numpy as np

# Function to calculate Mean Absolute Error (MAE)
def mean_absolute_error(true_values, noisy_values):
    return np.mean(np.abs(np.array(true_values) - np.array(noisy_values)))

# Function to calculate Variance of Noisy Averages
def variance(noisy_values):
    return np.var(noisy_values)

# Compute MAE for Local and Global DP
mae_local = mean_absolute_error(true_means, local_noisy_means)
mae_global = mean_absolute_error(true_means, global_noisy_means)

# Compute Variance for Local and Global DP
variance_local = variance(local_noisy_means)
variance_global = variance(global_noisy_means)

# Print Utility Metrics
print(f"📊 Utility Metrics:")
print(f"Mean Absolute Error (Local DP): {mae_local:.2f}")
print(f"Mean Absolute Error (Global DP): {mae_global:.2f}")
print(f"Variance (Local DP): {variance_local:.2f}")
print(f"Variance (Global DP): {variance_global:.2f}")
print("\n")  # Add space for clarity


# Function to compute noisy means for a given epsilon
def compute_noisy_means_local(epsilon):
    reporter = ClassReporter(path / "hogwarts_student_performance.csv", epsilon)
    return [
        reporter.mean_with_noise_local('Gryffindor'),
        reporter.mean_with_noise_local('Ravenclaw'),
        reporter.mean_with_noise_local('Slytherin'),
        reporter.mean_with_noise_local('Hufflepuff'),
    ]
    
# Function to compute noisy means for a given epsilon (GP)
def compute_noisy_means_global(epsilon):
    reporter = ClassReporter(path / "hogwarts_student_performance.csv", epsilon)
    return [
        reporter.mean_with_noise_global('Gryffindor'),
        reporter.mean_with_noise_global('Ravenclaw'),
        reporter.mean_with_noise_global('Slytherin'),
        reporter.mean_with_noise_global('Hufflepuff'),
    ]

# Privacy budgets to evaluate
epsilons = [0.1, .5, 1, 5, 10, 20]
local_mae = []
global_mae = []
local_epsilon_noise_pairs = {}
global_epsilon_noise_pairs = {}
# Generate 
for epsilon in epsilons:
    # Local DP
    reporter._epsilon = epsilon  # Update epsilon for local DP
    local_noisy_means = compute_noisy_means_local(epsilon)
    local_epsilon_noise_pairs[epsilon] = local_noisy_means
    local_mae.append(mean_absolute_error(true_means, local_noisy_means))

    # Global DP (already implemented similarly in `mean_with_noise`)
    reporter._epsilon = epsilon  # Update epsilon for global DP
    global_noisy_means = compute_noisy_means_global(epsilon)
    global_epsilon_noise_pairs[epsilon] = global_noisy_means
    global_mae.append(mean_absolute_error(true_means, global_noisy_means))

# Plot Privacy-Utility Tradeoff
import matplotlib.pyplot as plt

plt.plot(epsilons, local_mae, label='Local DP MAE', marker='o', color='orange')
plt.plot(epsilons, global_mae, label='Global DP MAE', marker='o', color='green')
plt.xlabel('Privacy Budget (ε)')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Privacy-Utility Tradeoff')
plt.legend()
plt.grid()
plt.show()


def membership_inference_attack(epsilon, dataset_file, record_index, dp_type='local'):
    """
    Simulates a membership inference attack by evaluating the noisy mean of a specific record's house,
    with and without that record included in the dataset. It then checks if the presence of the record 
    changes the noisy mean significantly, indicating membership.

    Args:
        epsilon (float): Privacy budget for the DP mechanism.
        dataset_file (str): Path to the dataset CSV file.
        record_index (int): Index of the record to perform the attack on.
        dp_type (str): Type of Differential Privacy mechanism to use ('local' or 'global').

    Returns:
        dict: Metrics for the membership inference attack (TPR, FPR, ASR).
    """
    # Load the dataset
    reporter = ClassReporter(dataset_file, epsilon)
    full_df = reporter._df

    # Extract the record to be analyzed
    record_to_attack = full_df.iloc[record_index]
    house = record_to_attack['house']
    
    # Remove the record from the dataset for the "non-member" case
    without_record_df = full_df.drop(record_index)
    reporter._df = without_record_df
    
    if dp_type == 'local':
        noisy_means_with = local_epsilon_noise_pairs[epsilon]
        
        if house == 'Gryffindor':
            noisy_mean_with = noisy_means_with[0]
            noisy_mean_without = reporter.mean_with_noise_local('Gryffindor')
        elif house == 'Ravenclaw':
            noisy_mean_with = noisy_means_with[1]
            noisy_mean_without = reporter.mean_with_noise_local('Ravenclaw')
        elif house == 'Slytherin':
            noisy_mean_with = noisy_means_with[2]
            noisy_mean_without = reporter.mean_with_noise_local('Slytherin')
        else:
            noisy_mean_with = noisy_means_with[3]
            noisy_mean_without = reporter.mean_with_noise_local('Hufflepuff')

    elif dp_type == 'global':
        noisy_means_with = global_epsilon_noise_pairs[epsilon]
        if house == 'Gryffindor':
            noisy_mean_with = noisy_means_with[0]
            noisy_mean_without = reporter.mean_with_noise_global('Gryffindor')
        elif house == 'Ravenclaw':
            noisy_mean_with = noisy_means_with[1]
            noisy_mean_without = reporter.mean_with_noise_global('Ravenclaw')
        elif house == 'Slytherin':
            noisy_mean_with = noisy_means_with[2]
            noisy_mean_without = reporter.mean_with_noise_global('Slytherin')
        else:
            noisy_mean_with = noisy_means_with[3]
            noisy_mean_without = reporter.mean_with_noise_global('Hufflepuff')
    else:
        raise ValueError("Invalid dp_type. Choose either 'local' or 'global'.")
    
    membership_inference = 1 if noisy_mean_with > noisy_mean_without else 0
    
    true_label = 1
    
    tpr = 1 if membership_inference == true_label else 0  # True Positive Rate (TPR)
    fpr = 1 if membership_inference != true_label else 0  # False Positive Rate (FPR)

    return {
        "True Positive Rate (TPR)": tpr,
        "False Positive Rate (FPR)": fpr,
        "Attack Success Rate (ASR)": (tpr + (1 - fpr)) / 2,
    }


# Example usage
# epsilon = {EPSILON}
# dataset_file = path / "hogwarts_student_performance.csv"

# print("Beginning local dp attacks...")
# # Attack for Local DP
# metrics_local = membership_inference_attack_half_split(
#     epsilon=epsilon,
#     dataset_file=dataset_file,
#     dp_type='local'
# )

# print("Beginning global dp attacks...")
# # Attack for Global DP
# metrics_global = membership_inference_attack_half_split(
#     epsilon=epsilon,
#     dataset_file=dataset_file,
#     dp_type='global'
# )

# Print results
# print("Local Differential Privacy Attack Metrics:")
# for metric, value in metrics_local.items():
#     print(f"{metric}: {value:.2f}")

# print("\nGlobal Differential Privacy Attack Metrics:")
# for metric, value in metrics_global.items():
#     print(f"{metric}: {value:.2f}")

import matplotlib.pyplot as plt

def evaluate_attack_across_epsilons(dataset_file, epsilon_values, record_index, dp_type='local'):
    """
    Evaluates the membership inference attack performance across a range of epsilon values
    for both Local and Global Differential Privacy.

    Args:
        dataset_file (str): Path to the dataset CSV file.
        epsilon_values (list): List of epsilon values for evaluation.
        dp_type (str): Type of Differential Privacy ('local' or 'global').

    Returns:
        dict: A dictionary with epsilon values as keys and attack metrics (TPR, FPR, ASR) as values.
    """
    results = {}

    for epsilon in epsilon_values:
        print(f"Running attack with ε = {epsilon} for {dp_type}...")
        metrics = membership_inference_attack(
            epsilon=epsilon,
            record_index=record_index,
            dataset_file=dataset_file,
            dp_type=dp_type
        )
        results[epsilon] = metrics

    return results

import matplotlib.pyplot as plt
import random

def visualize_attack_performance(dataset_file, num_members, epsilon_values):
    """
    Visualizes the attack performance (TPR, FPR, ASR) across a range of epsilon values
    for both Local and Global Differential Privacy on the same graph.

    Args:
        dataset_file (str): Path to the dataset CSV file.
        epsilon_values (list): List of epsilon values for evaluation.
        num_members (int): Number of random records to attack in the dataset.
    """
    
    # Initialize lists to store results for plotting
    tpr_values_local = {epsilon: [] for epsilon in epsilon_values}
    fpr_values_local = {epsilon: [] for epsilon in epsilon_values}
    asr_values_local = {epsilon: [] for epsilon in epsilon_values}
    
    tpr_values_global = {epsilon: [] for epsilon in epsilon_values}
    fpr_values_global = {epsilon: [] for epsilon in epsilon_values}
    asr_values_global = {epsilon: [] for epsilon in epsilon_values}
    
    # Generate random record indices
    record_indices = random.sample(range(len(reporter._df)), num_members)

    # Evaluate attack performance for Local DP
    for record_index in record_indices:
        results_local = evaluate_attack_across_epsilons(dataset_file, epsilon_values, record_index, dp_type='local')
        for epsilon in epsilon_values:
            metrics = results_local[epsilon]
            tpr_values_local[epsilon].append(metrics["True Positive Rate (TPR)"])
            fpr_values_local[epsilon].append(metrics["False Positive Rate (FPR)"])
            asr_values_local[epsilon].append(metrics["Attack Success Rate (ASR)"])

    # Evaluate attack performance for Global DP
    for record_index in record_indices:
        results_global = evaluate_attack_across_epsilons(dataset_file, epsilon_values, record_index, dp_type='global')
        for epsilon in epsilon_values:
            metrics = results_global[epsilon]
            tpr_values_global[epsilon].append(metrics["True Positive Rate (TPR)"])
            fpr_values_global[epsilon].append(metrics["False Positive Rate (FPR)"])
            asr_values_global[epsilon].append(metrics["Attack Success Rate (ASR)"])

    # Average the results across all records
    tpr_values_local_avg = [sum(tpr_values_local[epsilon]) / num_members for epsilon in epsilon_values]
    fpr_values_local_avg = [sum(fpr_values_local[epsilon]) / num_members for epsilon in epsilon_values]
    asr_values_local_avg = [sum(asr_values_local[epsilon]) / num_members for epsilon in epsilon_values]

    tpr_values_global_avg = [sum(tpr_values_global[epsilon]) / num_members for epsilon in epsilon_values]
    fpr_values_global_avg = [sum(fpr_values_global[epsilon]) / num_members for epsilon in epsilon_values]
    asr_values_global_avg = [sum(asr_values_global[epsilon]) / num_members for epsilon in epsilon_values]

    # Plot the results
    plt.figure(figsize=(10, 6))
    
    # Plot Local DP curves
    plt.plot(epsilon_values, tpr_values_local_avg, label='Local DP - TPR', marker='o', linestyle='-', color='blue')
    plt.plot(epsilon_values, fpr_values_local_avg, label='Local DP - FPR', marker='x', linestyle='--', color='blue')
    plt.plot(epsilon_values, asr_values_local_avg, label='Local DP - ASR', marker='^', linestyle='-.', color='blue')

    # Plot Global DP curves
    plt.plot(epsilon_values, tpr_values_global_avg, label='Global DP - TPR', marker='o', linestyle='-', color='green')
    plt.plot(epsilon_values, fpr_values_global_avg, label='Global DP - FPR', marker='x', linestyle='--', color='green')
    plt.plot(epsilon_values, asr_values_global_avg, label='Global DP - ASR', marker='^', linestyle='-.', color='green')

    plt.xlabel('Epsilon')
    plt.ylabel('Rate')
    plt.title("Attack Performance for Local and Global DP")
    plt.legend()
    plt.grid(True)
    plt.show()



print('visualizing attack performance')
# Visualize attack performance for both Local and Global DP
visualize_attack_performance(
    dataset_file=path / "hogwarts_student_performance.csv",
    num_members=50,
    epsilon_values=epsilons
)

