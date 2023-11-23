"""Parse output files from DFT simulations.

Scripts in this file are used to parse data output
and save it to a json like structure. We then save
each json like object to an ASE database.

Eventually, we would like to use the aims parser."""
import ase.io
import os
from numpy.linalg import norm
import logging
from absl import flags
from absl import app
import sys
import glob
import os
import sys
import csv
import numpy as np
import mendeleev
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import data_gen.gen_data as dg
import parsing.json_parser as js


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'paths_txt_file', 'None',
    'Where to find list of paths to parse.')
flags.DEFINE_string(
    'csv_filename',
    'None',
    'Where to store data as csv that has been parsed.')
flags.DEFINE_string(
    'db_name',
    'None',
    'Where to save parsed data as ASE db.')
flags.DEFINE_string(
    'paths_to_resubmit',
    'None',
    'Paths where have a nice day was not found.'
    'and paths that ran out of time.')
flags.DEFINE_string(
    'paths_misbehaving',
    'None',
    'Paths where have a nice day was not found'
    'and paths expired.')


class OutputParser():
    """Parses data output."""
    def __init__(
            self, paths_txt_file, csv_filename, db_name,
            paths_to_resubmit, paths_misbehaving,
            paths_increase_charge_mix,
            paths_decrease_charge_mix, paths_out_of_npl,
            paths_symmetry_issues):
        """Constructor

        Args:
        path_txt_file: (str) of paths where DFT output files
            can be found.
        csv_filename: (str) where to save data to a csv file.
        db_name: (str) where to save data as an ASE db.
        paths_to_resubmit: (str) path where to save simulation
            paths that didn't exit nicely so they can be
            resubmited.
        paths_misbehaving: path to files where we are not sure what went wrong
            but they didn't simulate correctly.
        paths_increase_charge_mix:
        paths_decrease_charge_mix:
        """
        # Normally a file containing paths on new lines is given
        # and not a list of paths.
        self.path_list = self.get_path_list(paths_txt_file)
        # self.atoms_obj_list = atoms_obj_list
        self.logger = logging.getLogger(__name__)
        # Get the atoms object from the path
        # self.read_geometry()
        # Create a DataGen Object to grab the tier map.
        self.tier_map = dg.GenData('None', 'None', 'None', 'None').tier_map
        # CSV filename of where to save parsed data.
        self.csv_filename = csv_filename
        self.db_name = db_name  # ASE db name.
        # Name of columns for header in csv.
        self.csv_columns = [
            'ml_architecture', 'dataset', 'batching_type', 'batch_size',
            'compute_device', 'iteration']

        # Define a list of paths that we shoudl resubmit
        # to a longer queue since they expired during calculation.
        self.paths_to_resubmit = paths_to_resubmit
        # We also define a list of paths that didn't exit well
        # and the reason is not that the time didn't expire.
        self.paths_misbehaving = paths_misbehaving


    def get_path_list(self, paths_txt_file):
        """Convert .txt file with paths in each newline to list.

        Args:
        path_txt_file: (string) path to a txt file where there
            is a path to a different simulation on each newline.

        Returns:
        path_list: (list) list of path strings to each simulation
            script.
        """
        with open(paths_txt_file, "r") as txt_file:
            # Strip new line chars (\n) from each line.
            path_list = [line.rstrip('\n') for line in txt_file]
        return path_list

    def create_header(self):
        """Create csv with header."""
        try:
            with open(self.csv_filename, 'w') as csvfile:
                writer = csv.DictWriter(
                    csvfile, fieldnames=self.csv_columns)
                # Write header.
                writer.writeheader()
        except IOError:
            print("I/O error")
            self.logger.error("I/O error writing header to csv file.")
            sys.exit('I/O issues writing header to csv file.')

    def read_geometry(self, path):
        """Collect data from geometry file.

        Args:
        path: to folder containing geometry.in file.
        """
        # Check if path to geometry.in file exists
        if relaxation:
            geometry_file = path + '/relaxation/geometry.in.next_step'
        else:
            geometry_file = path + '/geometry.in'
        if os.path.exists(geometry_file):
            atoms_object = ase.io.read(
                geometry_file, format='aims')
        else:
            if relaxation:
                print(
                    f'file {path}/relaxation/geometry.in.next_step does not exist.')
                self.logger.error(
                    f'file {path}/relaxation/geometry.in.next_step does not exist.')
            else:
                print(f'file {path}/geometry.in does not exist')
                self.logger.error('geometry input not found')
            sys.exit('geometry input not found.')

        return atoms_object

    def read_ase_params(self, path):
        """Read ASE Parameters used in the simulation."""
        ase_param_file = path + 'parameters.ase'
        if os.path.exists(path):
            params = ase.calculators.calculator.Parameters.read(
                ase_param_file)
        else:
            logging.error(
                'ASE Parameters file not found in %s', ase_param_file)
        return params


    def get_bandgap_data(self, calc):
        """"Get bandgap data from aims calculations."""
        if calc.name != 'aims':
            self.logging.error(
                'This method has not yet been developed for other codes.')
        # DTS: Initalize homo lumo gap to zero.
        HOMO_LUMO_gap = 0
        # Store the occupatancy values for each eigenvalue.
        # occupation = calc.get_occupations()  # Read occupation at Gamma point.
        # Initialize the bandgap to None.
        gap_bandstructure = None
        # DTS: we open out the aims.out file to read data.
        with open(calc.label + 'aims.out', 'r') as aims_output:
            # DTS: we go line by line in the output.out file.
            for line in aims_output.readlines():
                # Grab the aims version.
                if line.rfind('          Version ') != -1:
                    # Take last value of the line, the version.
                    aims_version = split_line(line)[-1]
                # Grab the estimated overall HOMO-LUMO gap.   
                if line.rfind('ESTIMATED overall HOMO-LUMO gap:') != -1:
                    HOMO_LUMO_gap = float(split_line(line)[4])  # VBM CBM Gap
                # Look for line showing energy differences between bands.
                if line.rfind('| Energy difference      :') != -1:
                    # DTS: if the last value of the occupation vector
                    # is non-zero then all energy levels are filled.
                    # This means we don't have a bandgap.
                    if occupation[-1] != 0:
                        gap_bandstructure = -1
                    else:
                        # DTS: Grab the bandstructure gap
                        if split_line(line)[-2] != '****************':
                            gap_bandstructure = float(split_line(line)[-2])

        if gap_bandstructure is None:
            # Check to see if all KS states are occupied.
            if occupation[-1] != 0:
                # If this is true then all states
                # are occupied and there is no orbital
                # that is unoccupied -> LUMO = 0.
                LUMO = 0
                HOMO = -1
            else:
                gap_bandstructure = self.get_band_file_data(calc)
        return HOMO_LUMO_gap, gap_bandstructure

    def get_band_file_data(self, calc):
        """Take a deep look into bandstructure files to get bandgap."""
        band_files = glob.glob(calc.label+'band*')
        gap_bandstructure = 1.0e6
        HOMO = -1.0e6
        LUMO = 1.0e6
        # Go through each band file one at a time.
        # Update the HOMO if we find a larger HOMO.
        # Update LUMO if we find a smaller LUMO.
        for band in band_files:
            # Previously laodtxt was not defined, so I assumed it's np.loadtext.
            data = np.loadtxt(band)[:, 4:]
            HOMO = max(
                HOMO,
                data[:, np.arange(1, len(data[0, :]), 2)][
                    data[:, np.arange(0, len(data[0, :]), 2)] != 0].max())
            if sum(data[:, np.arange(0, len(data[0, :]), 2)] == 0) != 0:
                LUMO = min(LUMO, data[:, np.arange(1, len(data[0, :]), 2)][
                    data[:, np.arange(0, len(data[0, :]), 2)] == 0].min())

        gap_bandstructure = LUMO-HOMO
        return gap_bandstructure

    @staticmethod
    def parse_path(path, ICSD_number=True, expansion=False):
        """Parse data from file path.

        This method takes in a path and splits
        the path based on forward slashes. It works
        backwards (right to left) and assigns values
        to what settings were used for data contained
        in the path.

        Args:
        path: (str) full pathname to folder where calc output is stored.

        Returns:
        settings_dict: (dict) dictionary that contains data to be saved.
        """
        # Split the path name based on forward slashes
        list_of_settings = path.split('/')
        setting_dict = {}
        try:
            if ICSD_number:
                setting_dict['compound_name'] = list_of_settings[-2].split('_')[0]
                setting_dict['ICSD_number'] = int(list_of_settings[-2].split('_')[-1])
            else:
                setting_dict['compound_name'] = list_of_settings[-2]
            setting_dict['k_point_density'] = int(list_of_settings[-3])
            setting_dict['rel_setting'] = list_of_settings[-4]
            setting_dict['basis_size'] = list_of_settings[-5]
            setting_dict['num_setting'] = list_of_settings[-6]
            setting_dict['functional'] = list_of_settings[-7]
            if expansion:
                setting_dict['expansion'] = list_of_settings[-8]
        except IndexError:
            # Static method doesn't can't access private member
            # variable.
            print('path: %s, is not properly formatted', path)
            sys.exit('path not formated correctly')
        return setting_dict

    def gather_all_path_data(self, submission_path):
        """Parse all data contained in path.

        Go to the path folder and get data from the
        files living there.

        Args:
        path: (str) path to the submission script that was
            used to submit simulation.

        Returns:
        data_dict: (dict) containing data that was parsed from
            files living in the path.
        ase_atoms_obj: (ASE Atoms Object) contains structural data
            used in the simulation. Useful for storing in an ASE db.        
        """
        # Save the time and day for later use to know when
        # a row was added to a database/csv.
        time_and_day = get_time_and_day()
        # Get information related to settings from the
        # submission script path. E.g. batch size, batching type.
        data_dict = self.parse_path(submission_path)
        # Ok, now remove the last part of the path since
        # the /submission_XY.sh is not useful. Let's take
        # the parent directory.
        parent_path = os.path.dirname(submission_path)
        # Check if the simulation was even started.


        # Check if sim finished, if not if time expired.
        calc_finished_bool, most_recent_error_file = self.check_calc_finished(
                parent_path)

        if calc_finished_bool:
            # Grab number of times it recompiled
		    # Grab timing infromation from the log
            self.get_recompile_and_timing_info(submission_path)
            # Add time/day when this row of data was grabbed.
            data_dict['time_day'] = time_and_day
            # Add the path from which data was taken.
            data_dict['path'] = submission_path

            data_dict = self.update_dict_with_batching_method_size(
                parent_path, data_dict)
            data_dict = self.grab_recompiles_and_timing(
                most_recent_error_file, data_dict)
            return data_dict

        else:
            print(f'path: {submission_path} added as misbehaving.')
            self.add_misbehaving_path(submission_path)
            return None
        
    def grab_recompiles_and_timing(self, most_recent_error_file, data_dict):
        """Go through err file and parse the recompilation and timing info.

        I1107 20:07:21.992192 22949931718464 train.py:72] LOG Message: Recompiling!

        07 20:59:07.525845 22949931718464 train.py:605] Step 1000000 train loss: 1.8053530084216618e-06
        I1107 20:59:07.705219 22949931718464 train.py:349] RMSE/MAE train: [0.0019891  0.00134341]
        I1107 20:59:07.705431 22949931718464 train.py:349] RMSE/MAE validation: [0.06648617 0.01503589]
        I1107 20:59:07.705571 22949931718464 train.py:349] RMSE/MAE test: [0.6293168  0.05498861]
        I1107 20:59:07.709884 22949931718464 train.py:623] Reached maximum number of steps without early stopping.
        I1107 20:59:07.710278 22949931718464 train.py:630] Lowest validation loss: 0.0664861650789436
        I1107 20:59:07.765833 22949931718464 train.py:633] Mean batching time: 0.0005979892456531525
        I1107 20:59:07.805134 22949931718464 train.py:636] Mean update time: 0.0026955132026672364
        """
        recompilation_counter = 0
        step_counter = 0
        step_200_000_train_loss = np.nan
        step_400_000_train_loss = np.nan
        step_600_000_train_loss = np.nan
        step_800_000_train_loss = np.nan
        step_1_000_000_train_loss = np.nan

        with open(most_recent_error_file, 'r') as fin:
            for line in list(fin.readlines()):
                if "LOG Message: Recompiling!" in line:
                    recompilation_counter = recompilation_counter + 1
                if "Step " in line:

                    split_line = line.split(' ')
                    step_num = split_line[-4]
                elif 'RMSE/MAE train' in line:
                    # Grab the training loss
                    rmse = line.split(' ')[-1].split(']')[-2]
                    data_dict[f'step_{step_num}_train_rmse'] = rmse
                elif 'RMSE/MAE validation' in line:
                    # Grab the training loss
                    rmse = line.split(' ')[-1].split(']')[-2]
                    data_dict[f'step_{step_num}_val_rmse'] = rmse                    
                elif 'RMSE/MAE test' in line:
                    # Grab the training loss
                    rmse = line.split(' ')[-1].split(']')[-2]
                    data_dict[f'step_{step_num}_test_rmse'] = rmse
                elif 'RMSE/MAE test' in line:
                    # Grab the training loss
                    rmse = line.split(' ')[-1].split(']')[-2]
                    data_dict[f'step_{step_num}_test_rmse'] = rmse
        
        data_dict['recompilation_counter'] = recompilation_counter
        return data_dict
        
    def update_dict_with_batching_method_size(self, parent_path, data_dict):
        """Parse information about the batching method and size.

        Sample path:
        /mpnn/aflow/dynamic/64/gpu_a100/iteration_5"""
        settings_list = parent_path.split('/')
        data_dict['iteration'] = settings_list[-1]
        data_dict['computing_type'] = settings_list[-2]
        data_dict['batch_size'] = settings_list[-3]
        data_dict['batching_type'] = settings_list[-4]
        data_dict['dataset'] = settings_list[-5]
        data_dict['model'] = settings_list[-6]
        return data_dict


        

    def get_recompile_and_timing_info(self):
        pass

    def check_if_djob_out_exists(self, submission_path):
        """Check if the simulation never ran."""
        if os.path.isfile(submission_path + '/relaxation/calculation/aims.out'):
            return False
        else:
            return True

    def get_queue(self, submission_path):
        """Get queue information from the submission path script.

        Args:
        submissions_path: (str) simulation submission script.

        Returns:
        queue: (str) which queue the script was submitted to -
            general/short."""
        queue = 'None'
        with open(submission_path) as f:
            script_text = f.read()
            if '#SBATCH --partition=short' in script_text:
                queue = 'short'
            elif '#SBATCH --partition=general' in script_text:
                queue = 'general'
            else:
                self.logger.error(
                    'Unable to find queue in sub'
                    'script %s' % submission_path)
                print('Unable to find queue info %s' % submission_path)
        return queue

    def add_expired_path(self, submission_path):
        """Add a path name where a simulation expired to due to being out of time.

        Args:
        path: (str) path name to submission script so that it
            can be resubmitted.
        """
        with open(
                self.paths_to_resubmit, 'a') as fo:
            fo.writelines(submission_path + '\n')

    def add_misbehaving_path(self, submission_path):
        """Add a path name where a simulation didn't end nicely.

        Args:
        path: (str) path name to submission script so that it
            can be resubmitted.
        """
        with open(
                self.paths_misbehaving, 'a') as fo:
            fo.writelines(submission_path + '\n')


    @staticmethod
    def check_calc_finished(parent_path):
        """Check if simulation exited nicely.

        We look to see if a .err file was created meaning the profiling
        experiment was at least started.

        Args:
            path: (str) path to folder containing error files.

        Returns:
            calc_finished_bool: (bool) True if calc exited
            nicely.
        """
        # Grab all .err files.

        error_files = glob.glob(parent_path + '/*.err').sort()
        # If the error_files list is empty, return calc_finished is false.
        
        if len(error_files) == 0: 
            calc_finished_bool = False
            most_recent_error_file = None
        else:
            calc_finished_bool = True
            most_recent_error_file = error_files[-1]


        return calc_finished_bool, most_recent_error_file

    @staticmethod
    def get_most_recent_djob_err(path):
        """Get the most recent job error path for a simulation.

        Args:
        path: (str) string to the submission script path.

        Returns:
        djob_path: (str) string to the most recent djob error path.
        """
        # Get list of files in the folder where djob.err would
        # live (aims.out as well).
        file_list = os.listdir(path)
        # Now we need to check for djob.err type files.
        djob_err_list = [x for x in file_list if 'djob.err.' in x]
        # Check if the list is empty
        if not djob_err_list:
            return None
        # Now we need to choose the most recent djob err file. Let's
        # look at which has the largest integer after splitting the
        # file name by . and looking at last value. We first
        # list of integers of name of Jobs.
        djob_int_list = [x.split('.')[-1] for x in djob_err_list]
        # Now find the index with the largest integer.
        most_recent_job_num = djob_int_list.index(max(djob_int_list))
        # Use the index to get the djob err path name.
        most_recent_djob_err_path = djob_err_list[most_recent_job_num]

        return most_recent_djob_err_path

    @staticmethod
    def check_sim_time_lim(path):
        """If time expired on sim, it returns True.

        If the sim didn't have a Have a nice day in aims.out
        then this method should be called to determine
        if the sim simply ran out of time because it was submitted
        to a queue that let it run only for a short amount of time.

        First we check for any file names with /djob.err.* in our folder.
        We do so by first getting a list of all files in our folder.
        Then looking if there's a match with the type. Then we choose
        the djob.err with the largest #. We look in this djob.err
        for markers CALCELLED and TIME LIMIT EXPIRED. If we find
        them we return true.

        Args:
        path: (str) path to folder containing aims.out.

        Returns:
        expired_time_bool: (bool) True if the sim ran out of time.
            False otherwise.
        """

        # Now go through each line in the most recent djob.err.*
        # file in the folder and see if we can spot the marker.
        first_marker = 'CANCELLED'
        second_marker = 'DUE TO TIME LIMIT'
        # By default the bool we return is False since we
        # haven't seen markers.
        expired_time_bool = False
        out_of_npl_bool = False
        # Get the path to most recent djob error.
        most_recent_djob_err = OutputParser.get_most_recent_djob_err(path)
        if most_recent_djob_err is not None:
            # Print path of most recent djob error.
            print('Most Recent Djob Err: %s' % most_recent_djob_err)
            for line in reversed(list(open(path + '/' + most_recent_djob_err))):
                print(line)
                if first_marker in line and second_marker not in line:
                    out_of_npl_bool = True
                if first_marker in line and second_marker in line:
                    expired_time_bool = True
                    break
        return expired_time_bool, out_of_npl_bool

    def connect_to_db(self):
        """Write simulation data to a ASE database.

        We go through each path in the list of paths.
        We then grab all data/settings into a dictionary
        and ASE Atoms object. We store the dict in a csv
        row and we store parts of the dict and ASE Atoms
        object into a row of the ASE database.
        """
        # Create header if csv file doesn't exist.
        if not os.path.isfile(self.csv_filename):
            self.create_header()
        with ase.db.core.connect(self.db_name) as db:
            with open(self.csv_filename, 'a') as csv_file:
                dict_writer = csv.DictWriter(
                    csv_file, fieldnames=self.csv_columns)
                self.write_all_path_data(
                    dict_writer, db)

    def write_all_path_data(self, dict_writer, db):
        """Write all data from path list to csv and db.

        Args:
        dict_writer: (csv.DictWriter() object) used as handle
            to write dictionary data to a row in an open CSV.
        db: (ASE db handle) db that has been opened and
            we can easily write to it.
        """
        for path in self.path_list:
            try:
                # Get data stored in files in the path folder.
                data_dict, _, relaxed_ase_atoms_obj = self.gather_all_path_data(
                    path)
                if data_dict is None:
                    raise ValueError(f'No data dict for path: {path}')

                # Commented out for speedup
                dict_writer.writerow(data_dict)


                # Feed the following dictionary key worded args
                # to db write.
                # expansion=expansion, basis_functions=tier, name=path,
                # functional=functional, basis_size=basis_size,
                # rel_setting=rel_setting,
                # k_point_density=k_point_density, total_energy=total_energy,
                # free_energy=free_energy, gap=HOMO_LUMO_gap,
                # gap_gamma=gamma_gap, gap_bandstructure=gap_bandstructure,
                # volume=volume)

                # Commented out for speedup
                db.write(
                    relaxed_ase_atoms_obj, attach_calculator=True,
                    **data_dict)
            except Exception as e:
                print("This path: %s has issues" % path)
                print("Error: %s" % e)
                self.logger.error(
                    "This path: %s gives this error: %s" % (path, e))


def get_data_from_csv(atom_num, functional, column):
    """Get data from csv file for functional, atom num, column.

    Note this method is copied from dataframe_prep. TODO: dts,
    combine this method and the one from dataframe prep into
    a single method and test it better.

    atom_num: atom number by # of protons.
    functional (string): lda or pbe?
    column (string): what column of data are we looking at from the
        the fhi-aims monomers database are we looking at. ex. 'EA_half'.
        Check the csv to find the names of the columns available.
    """
    # CSVs containing really tight FHI-aims data about
    # things like electron affinity, ionization potential
    # and radii.
    self.pbe_features_csv = (
        '/home/speckhard/Documents/theory/errorbar_project/'
        'error_modelling/modelling/data/descriptor_aims_data/'
        'really_tight_full_cut20_pbesol.csv')
    self.lda_features_csv = (
        '/home/speckhard/Documents/theory/errorbar_project/'
        'error_modelling/modelling/data/descriptor_aims_data/'
        'really_tight_full_cut20_pw-lda.csv')
    # Load descriptor csv files into dataframes.
    self.pbe_features_df = pd.read_csv(self.pbe_features_csv)
    self.lda_features_df = pd.read_csv(self.lda_features_csv)

    if functional == 'pbe':
        csv_val = float(self.pbe_features_df[
            self.pbe_features_df[
                'Atomic number'] == atom_num][column])
    elif functional == 'pw-lda':
        csv_val = float(self.lda_features_df[
            self.lda_features_df[
                'Atomic number'] == atom_num][column])
    else:
        raise BadFunctional(functional)

    return csv_val


def split_line(lines):
    """Split input line"""
    # Strip() removes leading and trailing whitespace
    # Then we split on the whitespace between words.
    # Store as a numpy array.
    line_array = np.array(lines.strip().split(' '))
    # Remove any elements that might be empty strings
    # in the array.
    line_vals = line_array[line_array != '']
    # Return the line values.
    return line_vals


def get_time_and_day():
    """Return the time and day now as string.

    Returns:
    time_day_str: (str) time and day in a string.
    """
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%H_%M_%S__%d_%m_%Y")
    return dt_string


def main(argv):
    """Main fxn to allow us to call this method directly."""
    paths_txt_file = FLAGS.paths_txt_file
    if paths_txt_file == 'None':
        sys.exit('paths txt file is None')
    print('paths txt file is %s' % paths_txt_file)
    csv_filename = FLAGS.csv_filename
    if csv_filename == 'None':
        sys.exit('csv filename name is None')
    print('csv filename is %s' % csv_filename)
    db_name = FLAGS.db_name
    if db_name == 'None':
        sys.exit('db name is None')
    print('db name is %s' % db_name)
    paths_to_resubmit = FLAGS.paths_to_resubmit
    if paths_to_resubmit == 'None':
        sys.exit('paths to resubmit is None')
    print(
        'File to store paths to'
        ' resubmit %s' % paths_to_resubmit)
    paths_misbehaving = FLAGS.paths_misbehaving
    if paths_misbehaving == 'None':
        sys.exit('paths misbehaving')
    print(
        'File to store paths to'
        ' resubmit %s' % paths_misbehaving)

    parse_obj = OutputParser(
        paths_txt_file=paths_txt_file,
        csv_filename=csv_filename,
        db_name=db_name,
        paths_to_resubmit=paths_to_resubmit,
        paths_misbehaving=paths_misbehaving)
    # Now submit all jobs
    parse_obj.connect_to_db()


if __name__ == '__main__':
    app.run(main)
