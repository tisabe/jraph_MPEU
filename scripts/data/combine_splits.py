import json

import ase.db
from tqdm import tqdm

from jraph_MPEU.input_pipeline import load_split_dict

workdir_super = 'results/aflow_x_mp/egap_ins/'
aflow_split = load_split_dict(workdir_super+"train_aflow")
mp_split = load_split_dict(workdir_super+"train_mp")

db = ase.db.connect("databases/aflow_x_matproj/graphs_12knn_vec.db")

splits = {}
for row in tqdm(db.select(source_file="databases/aflow_x_matproj/graphs_12knn_vec.db")):
    if row.old_id in aflow_split:
        splits[row.id] = aflow_split[row.old_id]

for row in tqdm(db.select(source_file="databases/matproj/mp2018_graphs_12knn_vec.db")):
    if row.old_id in mp_split:
        splits[row.id] = mp_split[row.old_id]

print(len(splits))

with open(workdir_super+"train_combined/combine_split.json", 'w', encoding="utf-8") as splits_file:
    json.dump(splits, splits_file, indent=4, separators=(',', ': '))
