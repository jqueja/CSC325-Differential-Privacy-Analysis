"""
 Important to note that python's float type has the same precision as the C++
 double.
"""

import sys  # isort:skip

sys.path.append("../pydp")  # isort:skip

# stdlib
import os
from pathlib import Path
import statistics as s
from typing import Union

# third party
import pandas as pd
import importlib
random = importlib.import_module('random')

# pydp absolute
import pydp as dp
from pydp.algorithms.laplacian import BoundedMean
from pydp.algorithms.laplacian import BoundedSum
from pydp.algorithms.laplacian import Count
from pydp.algorithms.laplacian import Max

'''
Epsilon (ε) is the most important parameter in differential privacy. 
It controls how much "privacy loss" is allowed when releasing information about a dataset.
Lower is more private

Sensitivity (Δ) measures the maximum change in the output (like the sum, mean, or count) 
that occurs if you change just one data point in the dataset.
'''
EPSILON = 10
SENSITIVITY = 1

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
    def mean_with_noise(self, house: str) -> float:

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
    def mean_with_noise(self, house: str) -> float:
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

reporter = ClassReporter(path / "hogwarts_student_performance.csv", 1)

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
local_gryffindor_noisy_mean = reporter.mean_with_noise('Gryffindor')
local_ravenclaw_noisy_mean = reporter.mean_with_noise('Ravenclaw')
local_slytherin_noisy_mean = reporter.mean_with_noise('Slytherin')
local_hufflepuff_noisy_mean = reporter.mean_with_noise('Hufflepuff')

print(f"🏠 Ravenclaw Noisy Avg:\t{local_ravenclaw_noisy_mean:.2f}")
print(f"🏠 Slytherin Noisy Avg:\t{local_slytherin_noisy_mean:.2f}")
print(f"🏠 Gryffindor Noisy Avg:\t{local_gryffindor_noisy_mean:.2f}")
print(f"🏠 Hufflepuff Noisy Avg:\t{local_hufflepuff_noisy_mean:.2f}")
print("\n")  # Add space for clarity


# Print the house grade averages with noise (Global Differential Privacy Applied)
print(f"📢 Global Differential Privacy (ε = {EPSILON}):")
global_gryffindor_noisy_mean = reporter.mean_with_noise('Gryffindor')
global_ravenclaw_noisy_mean = reporter.mean_with_noise('Ravenclaw')
global_slytherin_noisy_mean = reporter.mean_with_noise('Slytherin')
global_hufflepuff_noisy_mean = reporter.mean_with_noise('Hufflepuff')

print(f"🏠 Ravenclaw Global Noisy Avg:\t{global_ravenclaw_noisy_mean:.2f}")
print(f"🏠 Slytherin Global Noisy Avg:\t{global_slytherin_noisy_mean:.2f}")
print(f"🏠 Gryffindor Global Noisy Avg:\t{global_gryffindor_noisy_mean:.2f}")
print(f"🏠 Hufflepuff Global Noisy Avg:\t{global_hufflepuff_noisy_mean:.2f}")
print("\n")  # Add space for clarity


# Print final summary for comparison
print("📋 Summary of Averages")
print("───────────────────────────────────────────")
print(f"🏠 Ravenclaw True Avg: \t\t{true_ravenclaw_avg:.2f}")
print(f"🏠 Ravenclaw Local DP Avg: \t{local_ravenclaw_noisy_mean:.2f}")
print(f"🏠 Ravenclaw Global DP Avg: \t{global_ravenclaw_noisy_mean:.2f}")
print("\n")
print(f"🏠 Slytherin True Avg: \t\t{true_slytherin_avg:.2f}")
print(f"🏠 Slytherin Local DP Avg: \t{local_slytherin_noisy_mean:.2f}")
print(f"🏠 Slytherin Global DP Avg: \t{global_slytherin_noisy_mean:.2f}")
print("\n")
print(f"🏠 Gryffindor True Avg: \t\t{true_gryffindor_avg:.2f}")
print(f"🏠 Gryffindor Local DP Avg: \t{local_gryffindor_noisy_mean:.2f}")
print(f"🏠 Gryffindor Global DP Avg: \t{global_gryffindor_noisy_mean:.2f}")
print("\n")
print(f"🏠 Hufflepuff True Avg: \t\t{true_hufflepuff_avg:.2f}")
print(f"🏠 Hufflepuff Local DP Avg: \t{local_hufflepuff_noisy_mean:.2f}")
print(f"🏠 Hufflepuff Global DP Avg: \t{global_hufflepuff_noisy_mean:.2f}")
