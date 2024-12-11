'''
random_data.py - generates house average data with DATA_SIZE entries
'''

DATA_SIZE = 5000

import pandas as pd
import numpy as np
import importlib
random = importlib.import_module('random')

# Function to generate unique wizard names
def generate_wizard_names(num_names):
    first_names = ['Albus', 'Gellert', 'Minerva', 'Bellatrix', 'Severus', 'Sirius', 
                   'Remus', 'Molly', 'Arthur', 'Filius', 'Horace', 'Lucius', 
                   'Nymphadora', 'Kingsley', 'Percy', 'Tom', 'Rubeus', 'Helga', 
                   'Godric', 'Rowena', 'Salazar', 'Newt', 'Leta', 'Theseus', 
                   'Bathilda', 'Wulfric', 'Dilys', 'Bridget', 'Hepzibah', 'Wilhelmina']
    
    last_names = ['Dumbledore', 'Grindelwald', 'McGonagall', 'Lestrange', 'Snape', 'Black', 
                  'Lupin', 'Weasley', 'Flitwick', 'Slughorn', 'Malfoy', 'Tonks', 
                  'Shacklebolt', 'Crouch', 'Diggory', 'Lovegood', 'Riddle', 'Hooch', 
                  'Hagrid', 'Bones', 'Carrow', 'Crabbe', 'Goyle', 'Peverell', 
                  'Selwyn', 'Yaxley', 'Zabini', 'Greengrass', 'Parkinson', 'Pucey']

    all_names = [f"{first} {last}" for first in first_names for last in last_names]
    # Can only generate so many names, just use Wizard and some number
    if len(all_names) < num_names:
        additional_names = [f"Wizard{i}" for i in range(num_names - len(all_names))]
        all_names.extend(additional_names)
        
    random.shuffle(all_names)
    return all_names[:num_names]

# Number of students for the larger dataset
num_students = DATA_SIZE

# Generate unique wizard names
student_names = generate_wizard_names(num_students)

# Generate random houses
houses = np.random.choice(['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin'], num_students)

# Generate grades, seed number for reproducing datasets if needed
np.random.seed(42) 

# Generate random grades for each house according to the hierarchy
potions_grades = np.select(
    [
        (houses == 'Ravenclaw'), 
        (houses == 'Slytherin'), 
        (houses == 'Gryffindor'), 
        (houses == 'Hufflepuff')
    ],
    [
        np.random.randint(85, 101, num_students),  # Ravenclaw: 85-100
        np.random.randint(70, 96, num_students),   # Slytherin: 70-95
        np.random.randint(50, 91, num_students),   # Gryffindor: 50-90
        np.random.randint(40, 86, num_students)    # Hufflepuff: 40-85
    ]
)

herbology_grades = np.select(
    [
        (houses == 'Ravenclaw'), 
        (houses == 'Slytherin'), 
        (houses == 'Gryffindor'), 
        (houses == 'Hufflepuff')
    ],
    [
        np.random.randint(85, 101, num_students),  
        np.random.randint(70, 96, num_students),  
        np.random.randint(50, 91, num_students),  
        np.random.randint(40, 86, num_students)  
    ]
)

darkarts_grades = np.select(
    [
        (houses == 'Ravenclaw'), 
        (houses == 'Slytherin'), 
        (houses == 'Gryffindor'), 
        (houses == 'Hufflepuff')
    ],
    [
        np.random.randint(85, 101, num_students),  
        np.random.randint(70, 96, num_students),  
        np.random.randint(50, 91, num_students),  
        np.random.randint(40, 86, num_students)  
    ]
)

flying_grades = np.select(
    [
        (houses == 'Ravenclaw'), 
        (houses == 'Slytherin'), 
        (houses == 'Gryffindor'), 
        (houses == 'Hufflepuff')
    ],
    [
        np.random.randint(85, 101, num_students),  
        np.random.randint(70, 96, num_students),  
        np.random.randint(50, 91, num_students),  
        np.random.randint(40, 86, num_students)  
    ]
)

# Create the grades dictionary
grades = {
    'id': range(1, num_students + 1),
    'student_name': student_names,
    'house': houses,
    'potions': potions_grades,
    'herbology': herbology_grades,
    'darkarts': darkarts_grades,
    'flying': flying_grades
}

# Creating DataFrame
df_large = pd.DataFrame(grades)

# Save the larger dataset to a CSV file
csv_path_large = "scenarios/membershipinference/nonmember_hogwarts_student_performance.csv"
df_large.to_csv(csv_path_large, index=False)

print("Data Generated")