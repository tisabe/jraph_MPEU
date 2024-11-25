import ase.db
from tqdm import tqdm
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor

db_aflow = ase.db.connect("databases/aflow/graphs_12knn_vec.db")
print(len(db_aflow))
db_mp = ase.db.connect("databases/matproj/mp2018_graphs_12knn_vec.db")
print(len(db_mp))

with ase.db.connect("databases/aflow_x_matproj/graphs_12knn_vec.db") as db_out:
    for row in tqdm(db_aflow.select()):
        atoms = row.toatoms()
        struct_pmg = AseAtomsAdaptor.get_structure(atoms)
        analyzer = SpacegroupAnalyzer(struct_pmg, symprec=0.1)
        spg = analyzer.get_space_group_number()
        db_out.write(
            atoms, row.key_value_pairs, row.data, old_id=row.id,
            source_file="databases/aflow_x_matproj/graphs_12knn_vec.db",
            band_gap=row.Egap, ef_atom=row.enthalpy_formation_atom)
    for row in tqdm(db_mp.select()):
        atoms = row.toatoms()
        struct_pmg = AseAtomsAdaptor.get_structure(atoms)
        analyzer = SpacegroupAnalyzer(struct_pmg, symprec=0.1)
        spg = analyzer.get_space_group_number()
        db_out.write(
            atoms, row.key_value_pairs, row.data, old_id=row.id,
            source_file="databases/matproj/mp2018_graphs_12knn_vec.db",
            ef_atom=row.delta_e, dft_type="['PAW_PBE']")

print(len(db_out))
