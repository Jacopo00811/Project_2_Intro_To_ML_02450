import numpy as np

with open("Project_2\\Project_2_Intro_To_ML_02450\\dataAvg.txt", "r") as file:
    line = file.read().strip().strip('[]').replace(',', '')
    values = np.array([float(value) for value in line.split()])

average = np.mean(values)
print("Average:", average)
