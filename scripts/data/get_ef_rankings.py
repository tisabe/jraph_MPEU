"""Get correct formation energy rankings for structures with same chem formula."""

import pandas as pd


def get_correct_min_ef_predicted(df):
    col_pred = 'enthalpy_formation_atom_predicted'
    for split in ['test']:  # TODO fix this.
            print(f'Split: {split}')
            df_split = df[df['split'] == split]
            print('Num rows: ', len(df_split))
            grouped = df_split.groupby('formula')
            # filter out groups with only one row/formulas that appear only once
            df_split = grouped.filter(lambda x: len(x) > 1)

            df_split = df_split.sort_values('enthalpy_formation_atom')
            # re-group since the filtering split up the groups
            grouped = df_split.groupby('formula')
            df_true_min = grouped[['auid', 'formula']].aggregate('first')
            auids_true = set(df_true_min['auid'].to_list())
            
            df_split = df_split.sort_values(col_pred)
            grouped = df_split.groupby('formula')
            df_pred_min = grouped[['auid', 'formula']].aggregate('first')
            auids_pred = set(df_pred_min['auid'].to_list())
            print(len(auids_true.intersection(auids_pred)), '/', len(auids_true))
            return len(auids_true.intersection(auids_pred))

# if __name__ == "__main__":
#     df = pd.read_csv()
#     main(df)