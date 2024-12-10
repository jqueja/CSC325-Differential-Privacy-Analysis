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

# pydp absolute
import pydp as dp
from pydp.algorithms.laplacian import BoundedMean
from pydp.algorithms.laplacian import BoundedSum
from pydp.algorithms.laplacian import Count
from pydp.algorithms.laplacian import Max


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
    

    '''
    def count_above(self, limit: int) -> int:
        return self._df[self._df.hours_worked > limit].count().iloc[4]

    # Function to calculate maximum number of hours worked in the column.
    def max(self) -> int:
        return self._df.max().iloc[4]

    # Function to return the remaining privacy budget.
    def privacy_budget(self) -> float:
        return self._privacy_budget

    # Function to return the DP sum of all hours worked.
    def private_sum(self, privacy_budget: float) -> float:
        x = BoundedSum(
            epsilon=privacy_budget,
            delta=0,
            lower_bound=0,
            upper_bound=100,
            dtype="float",
        )
        return x.quick_result(list(self._df["hours_worked"]))

    # Function to return the DP mean of all hours worked.
    def private_mean(self, privacy_budget: float) -> float:
        x = BoundedMean(
            epsilon=privacy_budget, lower_bound=0, upper_bound=100, dtype="float"
        )
        return x.quick_result(list(self._df["hours_worked"]))

    # Function to return the DP count of the number of elves who worked more than "limit" hours.
    def private_count_above(
        self, privacy_budget: float, limit: int
    ) -> Union[int, float]:
        x = Count(epsilon=privacy_budget, dtype="int")
        return x.quick_result(
            list(self._df[self._df.hours_worked > limit]["hours_worked"])
        )

    # Function to return the DP maximum of the number of carrots eaten by any one animal.
    def private_max(self, privacy_budget: float) -> Union[int, float]:
        # 0 and 150 are the upper and lower limits for the search bound.
        x = Max(epsilon=privacy_budget, lower_bound=0, upper_bound=100, dtype="int")
        return x.quick_result(list(self._df["hours_worked"]))
    '''


# get absolute path
path = Path(os.path.dirname(os.path.abspath(__file__)))

reporter = ClassReporter(path / "hogwarts_student_performance.csv", 1)

print("Total Students:\t" + str(reporter.sum_hours()))

# Print the house grade average
print(f"Ravenclaw Avg:\t{reporter.mean_ravenclaw_grades():.2f}")
print(f"Slytherin Avg:\t{reporter.mean_slytherin_grades():.2f}")
print(f"Gryffindor Avg:\t{reporter.mean_gryffindor_grades():.2f}")
print(f"Hufflepuff Avg:\t{reporter.mean_hufflepuff_grades():.2f}")
