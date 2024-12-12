# 🧙‍♀️ Data Analysis Project for Hogwarts Students 🧙‍♂️ 

## 🦉 Overview 🦉
At the Ministry, we are conducting a data analysis project tailored for the students of Hogwarts. This initiative uses differential privacy to safeguard individual privacy while uncovering meaningful trends and insights. By introducing controlled noise into the results, we ensure that no specific student's information can be identified, fostering trust and maintaining confidentiality. This method strikes a balance between protecting privacy and preserving data utility, enabling the secure sharing of aggregated insights.

## 📚 Table of Contents
- [🦉 Overview](#-overview-)
- [🤔 What is Differential Privacy?](#-what-is-differential-privacy-)
- [⚡️ Installation](#installation)
- [🖼️ Screenshots](#screenshots)
- [⚡️ Run Program](#run-program-depending-on-scenario)

### 🤔 What is Differential Privacy? 🤔

Differential privacy is a rigorous mathematical definition of privacy for statistical analysis and machine learning. In the simplest setting, consider an algorithm that analyzes a dataset and releases statistics about it (such as means and variances, cross-tabulations, or the parameters of a machine learning model). Such an algorithm is said to be differentially private if by looking at the output, one cannot tell whether any individual's data was included in the original dataset or not.

For more information on OpenDP, a popular open-source library for implementing differential privacy, visit the [OpenDP project page](https://opendp.org/about#:~:text=Differential%20privacy%20is%20a%20rigorous,of%20a%20machine%20learning%20model).).

### Scenario 1: 🧝‍♂️ House Elf Working Conditions 🧝‍♀️

**Use Case:** Surveying house elves anonymously about working conditions at Hogwarts.  

**Challenge:** House elves fear their masters discovering critical feedback about working hours or conditions.  

**Solution:** Use Differential Privacy (DP) to aggregate survey results.  

### Scenario 2: 🦁 Student Performace Analysis Across Houses 🐍

**Use Case:** Comparing academic performances across different houses.  

**Challenge:** Students might fear that their house reputation could be negatively impacted if data about lower-performing houses is released.  

**Solution:** Implement differential privacy to release only aggregated data on overall performances across houses. This way, the system can show trends without revealing individual student results, maintaining house loyalty and preventing possible stigma associated with poor performance in a particular house.  


## ⚡️ Installation ⚡️

To get started with the project, you'll need to set up a virtual environment and install the
required dependencies. Follow the steps below to set up your environment.

WARNING: Apple Silicon does not work. x86 machines only. ⚠️

### 1. 🧪 Create a Virtual Environment 🧪

Ensure that you have Python 3.7+ installed on your system.
```bash
python3 -m venv venv
```

### 2. 🔮 Activate Virtual Environment 🔮

```bash
source /venv/bin/activate
```
You should notice a change in your terminal prompt. It should now display (venv) at the beginning, indicating that the virtual environment is active.

### 3. 📜 Install Dependencies 📜
```bash
pip install -r requirements.txt
```

## Run Program depending on scenario
```bash
python3 elfhours.py
```
```bash
python3 houseavg.py
```
```bash
python3 membership_attack.py
```
