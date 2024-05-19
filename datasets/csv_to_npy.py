import numpy as np
import csv
import os

def csv_to_npy(csv_file, npy_file):
    # Read CSV file into a list of lists
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        data_list = [row for row in csv_reader]

    # Convert the list of lists to a NumPy array
    data_array = np.array(data_list)[1:]

    # Save the NumPy array to a .npy file
    np.save(npy_file, data_array)

def lucas_csv_to_npy():
    # Replace 'input.csv' with your CSV file and 'output.npy' with the desired output file name

    current_dir = os.getcwd()
    csv_relative = os.path.join('datasets', 'LUCAS.csv')
    csv_absolute = os.path.join(current_dir, csv_relative)

    npy_relative = os.path.join('datasets', 'LUCAS.npy')
    npy_absolute = os.path.join(current_dir, npy_relative)
    csv_to_npy(csv_absolute, npy_absolute)

    data = {
        0: "Smoking",
        1: "Yellow_Fingers",
        2: "Anxiety",
        3: "Peer_Pressure",
        4: "Genetics",
        5: "Attention_Disorder",
        6: "Born_an_Even_Day",
        7: "Car_Accident",
        8: "Fatigue",
        9: "Allergy",
        10: "Coughing",
        11: "Lung_Cancer"
    }

    adjmatrix = np.zeros((12, 12))
    adjmatrix[0][1] = 1
    adjmatrix[0][11] = 1
    adjmatrix[2][0] = 1
    adjmatrix[3][0] = 1
    adjmatrix[4][11] = 1
    adjmatrix[4][5] = 1
    adjmatrix[5][7] = 1
    adjmatrix[8][7] = 1
    adjmatrix[9][10] = 1
    adjmatrix[10][8] = 1
    adjmatrix[11][10] = 1
    adjmatrix[11][8] = 1

    sol_relative = os.path.join('datasets', 'LUCAS_sol.npy')
    sol_absolute = os.path.join(current_dir, sol_relative)
    np.save(sol_absolute, adjmatrix)

def asia_csv_to_npy():
    # Replace 'input.csv' with your CSV file and 'output.npy' with the desired output file name

    current_dir = os.getcwd()
    csv_relative = os.path.join('datasets', 'Asia.csv')
    csv_absolute = os.path.join(current_dir, csv_relative)

    npy_relative = os.path.join('datasets', 'Asia.npy')
    npy_absolute = os.path.join(current_dir, npy_relative)
    csv_to_npy(csv_absolute, npy_absolute)

    data = {
        0: 'VisitAsia', 
        1: 'TB', 
        2: 'Smoking', 
        3: 'Cancer', 
        4: 'TBorCancer', 
        5: 'XRay', 
        6: 'Bronchitis', 
        7: 'Dysphenia'
    }

    adjmatrix = np.zeros((8, 8))
    adjmatrix[0][1] = 1
    adjmatrix[1][4] = 1
    adjmatrix[2][3] = 1
    adjmatrix[2][6] = 1
    adjmatrix[3][4] = 1
    adjmatrix[4][5] = 1
    adjmatrix[4][7] = 1
    adjmatrix[6][7] = 1

    

    sol_relative = os.path.join('datasets', 'Asia_sol.npy')
    sol_absolute = os.path.join(current_dir, sol_relative)
    np.save(sol_absolute, adjmatrix)


def SACHS_csv_to_npy():
    # Replace 'input.csv' with your CSV file and 'output.npy' with the desired output file name

    current_dir = os.getcwd()
    csv_relative = os.path.join('datasets', 'SACHS.csv')
    csv_absolute = os.path.join(current_dir, csv_relative)

    npy_relative = os.path.join('datasets', 'SACHS.npy')
    npy_absolute = os.path.join(current_dir, npy_relative)
    csv_to_npy(csv_absolute, npy_absolute)

    data = {
        0: 'praf',
        1: 'pmek',
        2: 'plcg',
        3: 'PIP2',
        4: 'PIP3',
        5: 'p44/42',
        6: 'pakts473',
        7: 'PKA',
        8: 'PKC',
        9: 'P38',
        10: 'pjnk'
    }

    adjmatrix = np.zeros((11, 11))
    adjmatrix[8][1] = 1
    adjmatrix[8][0] = 1
    adjmatrix[8][7] = 1
    adjmatrix[8][10] = 1
    adjmatrix[8][9] = 1
    adjmatrix[7][0] = 1
    adjmatrix[7][1] = 1
    adjmatrix[7][5] = 1
    adjmatrix[7][6] = 1
    adjmatrix[7][10] = 1
    adjmatrix[7][9] = 1

    adjmatrix[0][1] = 1
    adjmatrix[1][5] = 1
    adjmatrix[5][6] = 1
    adjmatrix[2][3] = 1
    adjmatrix[2][4] = 1
    adjmatrix[4][3] = 1

    

    sol_relative = os.path.join('datasets', 'SACHS_sol.npy')
    sol_absolute = os.path.join(current_dir, sol_relative)
    np.save(sol_absolute, adjmatrix)



if __name__ == "__main__":
    SACHS_csv_to_npy()
    

