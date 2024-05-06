"""Prepare the data for the ElemNet model."""


from absl import flags
from absl import app

import ast
import pandas as pd
import numpy as np

# INPUT_CSV_FILENAME = "/home/dts/Documents/hu/jraph_MPEU/other_models/band_gap_data/result_best_egap_model_tim.csv"
INPUT_CSV_FILENAME = "/home/dts/Documents/hu/jraph_MPEU/other_models/band_gap_data/result_ensemble_ef.csv"
OUTPUT_CSV_FILENAME = "/home/dts/Documents/hu/jraph_MPEU/other_models/band_gap_data/elemnet_data_ef.csv"


def main(argv):

    max_num = 0

    df = pd.read_csv(INPUT_CSV_FILENAME)

    for numbers_list in df['numbers']:
        for atomic_number in numbers_list.strip('[').strip(']').split():
            if int(atomic_number) > max_num:
                print(f'this atomic number: {atomic_number} is bigger')
                max_num = int(atomic_number)

    # Ok we should one hot encode the atomic numbers. Why not use tim's code.
    # Tim's code works for a GNN where each node has a single atom.
    # Let's instead one hot encode each material.
    ohe_list_of_lists = []

    for numbers_list in df['numbers']:
        ohe_atomic_numbers = np.zeros(max_num)
        for atomic_number in numbers_list.strip('[').strip(']').split():
            ohe_atomic_numbers[int(atomic_number)-1] = 1
        # Now normalize the list to be max one.
        ohe_atomic_numbers = ohe_atomic_numbers / np.linalg.norm(ohe_atomic_numbers)
        ohe_list_of_lists.append(list(ohe_atomic_numbers))
    
    print(ohe_list_of_lists[0])
    print(np.linalg.norm(ohe_list_of_lists[0]))

    df['ohe_atomic_numbers'] = ohe_list_of_lists
    df.to_csv(OUTPUT_CSV_FILENAME, index=False)


if __name__ == '__main__':
    app.run(main)
