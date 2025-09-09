import json, os
from argparse import ArgumentParser
from tqdm import tqdm
from scipy.sparse import coo_matrix
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model


def create_doc_kp_matrix(phrase_occurrences):
    # first create phrase vocab
    phrase_vocab = list(sorted(phrase_occurrences.keys()))
    phrase2idx = {p:i for i, p in enumerate(phrase_vocab)}

    row, col, data = [], [], []

    for phrase_idx, phrase in tqdm(enumerate(phrase_vocab), desc = "Creating sparse matrix"):
        doc_ids = phrase_occurrences[phrase]
        row.extend(doc_ids)
        col.extend([phrase_idx] * len(doc_ids))
        data.extend([1] * len(doc_ids))

    n_rows = max(row) + 1
    n_cols = max(col) + 1

    sparse_mat = coo_matrix((data, (row, col)), shape=(n_rows, n_cols))

    return sparse_mat.tocsr(), phrase2idx


def optimization_problem(doc_kp_matrix, phrase2idx):
    model = cp_model.CpModel()

    num_j = len(phrase2idx)
    num_i = doc_kp_matrix.shape[0]
    y = [model.NewBoolVar(f'y_{j}') for j in range(num_j)]
    z = [model.NewBoolVar(f'z_{i}') for i in range(num_i)]

    for i in range(num_i):
        x_i = doc_kp_matrix[i]
        
        # Create weighted sum expression: S_i = sum(x_ij * y_j)
        S_i = sum(x_i[j] * y[j] for j in range(num_j))
        
        # If z_i is True, enforce S_i >= 1
        model.Add(S_i >= 1).OnlyEnforceIf(z[i])
        
        # If z_i is False, enforce S_i == 0
        model.Add(S_i <= 0).OnlyEnforceIf(z[i].Not())

    # Optional: Add objective to maximize number of active z_i
    model.Maximize(sum(z))

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # Print results
    if status == cp_model.OPTIMAL:
        print('Solution found')
        res = []
        for j in range(num_j):
            res.append(solver.Value(y[j]))

        number_of_documents_covered = 0
        for i in range(num_i):
            number_of_documents_covered += solver.Value(z[i])
        print("Number of documents covered", number_of_documents_covered)
    else:
        print('No solution found.')




def main():
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type = str, help = "Phrase occurrences folder")
    parser.add_argument("--num_phrases", type = int, default = 30000)

    args = parser.parse_args()

    input_folder = args.input_folder
    num_phrases = args.num_phrases

    input_files = os.listdir(input_folder)
    input_files = [os.path.join(input_folder, file) for file in input_files]


    phrase_occurrences = {}
    for file in input_files:
        with open(file) as f:
            temp = json.load(f)

            for k in temp:
                if k not in phrase_occurrences: phrase_occurrences[k] = []
                phrase_occurrences[k].extend(temp[k])


    doc_kp_matrix, phrase2idx = create_doc_kp_matrix()
    