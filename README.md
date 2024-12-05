# Data Analysis Project for Hogwarts Students

## 🦉Overview 🦉
At the Ministry, we are conducting a data analysis project tailored for the students of Hogwarts. This initiative uses differential privacy to safeguard individual privacy while uncovering meaningful trends and insights. By introducing controlled noise into the results, we ensure that no specific student's information can be identified, fostering trust and maintaining confidentiality. This method strikes a balance between protecting privacy and preserving data utility, enabling the secure sharing of aggregated insights.

Scenario 1: 🧝‍♂️ House Elf Working Conditions 🧝‍♀️
Use Case: Surveying house elves anonymously about working conditions at Hogwarts.
Challenge: House elves fear their masters discovering critical feedback about working hours or conditions.
Solution: Use Differential Privacy (DP) to aggregate survey results.

## ⚡️Installation ⚡️

To get started with the project, you'll need to set up a virtual environment and install the
required dependencies. Follow the steps below to set up your environment.

WARNING: Apple Silicon does not work. x86 machines only. 

### 1. 🧪 Create a Virtual Environment 🧪

Ensure that you have Python 3.7+ installed on your system.
```bash
python3 -m venv venv
```

### 2. 🔮Activate Virtual Environment 🔮

```bash
source /venv/bin/activate
```
You should notice a change in your terminal prompt. It should now display (venv) at the beginning, indicating that the virtual environment is active.

### 3. 📜Install Dependencies 📜
```bash
pip install pandas
```
```bash
pip install python-dp
```

## Run Program
```bash
python3 elfhours.py
```
