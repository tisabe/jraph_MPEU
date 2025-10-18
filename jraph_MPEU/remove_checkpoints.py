import os
import sys
import shutil

from absl import app
from absl import flags

# Define a flag for the job script file path
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'job_file',
    None,
    'Path to the text file containing a list of job script paths.'
)

# You can add more flags if needed, e.g., a dry_run flag
flags.DEFINE_boolean(
    'dry_run',
    True,
    'If True, only print what would be deleted without actually deleting anything.'
)

def delete_checkpoints_folders(job_file_path, dry_run=True):
    """
    Reads a file containing job script paths, identifies the parent directory
    of each script, and deletes the 'checkpoints' folder within that parent.

    Args:
        job_file_path (str): The path to the text file listing job scripts.
        dry_run (bool): If True, only print deletion actions; otherwise, perform actual deletion.
    """
    deleted_count = 0
    not_found_count = 0

    if not os.path.exists(job_file_path):
        print(f"Error: Job file '{job_file_path}' not found.", file=sys.stderr)
        return deleted_count, not_found_count

    print(f"Processing job scripts from: {job_file_path}")
    if dry_run:
        print("--- DRY RUN MODE: No files will be actually deleted. ---")
    else:
        print("--- REAL DELETION MODE: Files WILL be deleted. ---")



    with open(job_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            job_script_path = line.strip() # Remove leading/trailing whitespace

            if not job_script_path: # Skip empty lines
                continue

            # Get the directory of the job script
            job_script_dir = os.path.dirname(job_script_path)

            # Construct the path to the 'checkpoints' folder
            checkpoints_folder = os.path.join(job_script_dir, 'checkpoints')

            if os.path.isdir(checkpoints_folder):
                if dry_run:
                    print(f"[{line_num}] DRY RUN: Would delete: {checkpoints_folder}")
                else:
                    print(f"[{line_num}] Deleting: {checkpoints_folder}")
                    try:
                        shutil.rmtree(checkpoints_folder)
                        deleted_count += 1
                    except OSError as e:
                        print(f"Error deleting {checkpoints_folder}: {e}", file=sys.stderr)
            else:
                print(f"[{line_num}] Checkpoints folder not found: {checkpoints_folder}")
                not_found_count += 1

    print("\nDeletion process complete.")
    print(f"Total folders deleted (or would be deleted in dry run): {deleted_count}")
    print(f"Total folders not found: {not_found_count}")
    return deleted_count, not_found_count


def main(argv):
    """Main function to parse flags and run the deletion."""
    # argv will contain the script name and any unparsed arguments
    # absl.app.run() handles parsing known flags, so we don't need to manually parse FLAGS here.
    if FLAGS.job_file is None:
        print("Error: The --job_file flag is required.", file=sys.stderr)
        # You could also uncomment the line below to show help and exit
        # app.usage()
        sys.exit(1)

    delete_checkpoints_folders(FLAGS.job_file, dry_run=FLAGS.dry_run)


if __name__ == "__main__":
    app.run(main)