import pandas as pd
import numpy as np

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
    if len(all_names) < num_names:
        additional_names = [f"Wizard{i}" for i in range(num_names - len(all_names))]
        all_names.extend(additional_names)
        
    random.shuffle(all_names)
    return all_names[:num_names]

# Number of students for the larger dataset
num_students = 5000

# Generate unique wizard names
student_names = generate_wizard_names(num_students)

# Generate random houses
houses = np.random.choice(['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin'], num_students)

# Generate grades, seed number for reproducing datasets if needed
np.random.seed(42) 
grades = {
    'id': range(1, num_students + 1),
    'student name': student_names,
    'House': houses,
    'Potions grade': np.random.randint(40, 101, num_students),
    'Herbology grade': np.random.randint(40, 101, num_students),
    'Defense of the Dark Arts grade': np.random.randint(50, 101, num_students),
    'Flying class grade': np.random.randint(40, 101, num_students)
}

# Creating DataFrame
df_large = pd.DataFrame(grades)

# Save the larger dataset to a CSV file
csv_path_large = "hogwarts_student_performance"
df_large.to_csv(csv_path_large, index=False)

csv_path_large
