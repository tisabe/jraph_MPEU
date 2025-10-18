import os
import shutil
import pytest

from absl import flags

from jraph_MPEU import remove_checkpoints

# Define a flag for the job script file path
@pytest.fixture
def setup_test_environment(tmp_path):
    """
    A pytest fixture to create a temporary, isolated directory structure
    for each test.
    `tmp_path` is a built-in pytest fixture for creating a temporary directory.
    """
    # Define paths relative to the temporary root
    base_path = tmp_path / "test_root"

    # --- Scenario 1: Checkpoints to be deleted ---
    # Paths that match your example
    (base_path / "u/dansp/batching_steps/round_False/64/gpu_a100/iteration_7").mkdir(parents=True, exist_ok=True)
    (base_path / "u/dansp/batching_steps/round_False/64/gpu_a100/iteration_8").mkdir(parents=True, exist_ok=True)
    (base_path / "u/dansp/batching_steps/round_False/128/gpu_a100/iteration_0").mkdir(parents=True, exist_ok=True)
    (base_path / "u/dansp/batching_steps/round_False/128/gpu_a100/iteration_1").mkdir(parents=True, exist_ok=True) # Exists but no checkpoints in its parent

    (base_path / "u/dansp/batching_steps/round_False/64/gpu_a100/iteration_7/checkpoints").mkdir(parents=True, exist_ok=True)
    (base_path / "u/dansp/batching_steps/round_False/64/gpu_a100/iteration_8/checkpoints").mkdir(parents=True, exist_ok=True)
    (base_path / "u/dansp/batching_steps/round_False/128/gpu_a100/iteration_0/checkpoints").mkdir(parents=True, exist_ok=True)
    (base_path / "u/dansp/batching_steps/round_False/128/gpu_a100/iteration_1/checkpoints").mkdir(parents=True, exist_ok=True) # Exists but no checkpoints in its parent

    checkpoints_64_path = base_path / "u/dansp/batching_steps/round_False/64/gpu_a100/iteration_8/checkpoints"
    checkpoints_128_path = base_path / "u/dansp/batching_steps/round_False/128/gpu_a100/iteration_0/checkpoints"

    # Add some dummy files
    (checkpoints_64_path / "file1.txt").write_text("checkpoint_data_64")
    (checkpoints_128_path / "config.json").write_text("checkpoint_data_128")

    # Create dummy job script files
    (base_path / "u/dansp/batching_steps/round_False/64/gpu_a100/iteration_7/profiling_job.sh").touch()
    (base_path / "u/dansp/batching_steps/round_False/64/gpu_a100/iteration_8/profiling_job.sh").touch()
    (base_path / "u/dansp/batching_steps/round_False/128/gpu_a100/iteration_0/profiling_job.sh").touch()
    (base_path / "u/dansp/batching_steps/round_False/128/gpu_a100/iteration_1/profiling_job.sh").touch()


    # --- Scenario 2: Checkpoints in the wrong place or not present ---
    # This path's corresponding parent should NOT have a checkpoints folder to delete
    (base_path / "a/different/path/job_dir").mkdir(parents=True, exist_ok=True)
    (base_path / "a/different/path/job_dir/script.sh").touch()

    # This path has a 'checkpoints' folder, but it's *within* the parents's directory, not its parent's parent
    (base_path / "another/job/checkpoints/subfolder").mkdir(parents=True, exist_ok=True)
    (base_path / "another/job/path/").mkdir(parents=True, exist_ok=True)
    (base_path / "another/job/path/script.sh").touch()

    # A path that refers to a non-existent job script (and thus no checkpoints in its parent)
    # The parent directory `non/existent/base/dir/job_folder` won't be created as it's not relevant for this test.


    # --- Create the your_jobs.txt file ---
    job_file_content = f"""{base_path}/u/dansp/batching_steps/round_False/64/gpu_a100/iteration_7/profiling_job.sh
{base_path}/u/dansp/batching_steps/round_False/64/gpu_a100/iteration_8/profiling_job.sh
{base_path}/u/dansp/batching_steps/round_False/128/gpu_a100/iteration_0/profiling_job.sh
{base_path}/u/dansp/batching_steps/round_False/128/gpu_a100/iteration_1/profiling_job.sh
{base_path}/a/different/path/job_dir/script.sh
{base_path}/another/job/path/script.sh
/non/existent/base/dir/job_folder/test.sh
"""
    job_file_path = tmp_path / "your_jobs.txt"
    job_file_path.write_text(job_file_content)

    # Yield the relevant paths back to the test function
    yield {
        "job_file_path": str(job_file_path),
        "checkpoints_64_path": str(checkpoints_64_path),
        "checkpoints_128_path": str(checkpoints_128_path),
        "wrong_checkpoints_path": str(base_path / "another/job/checkpoints"),
        "tmp_path": tmp_path # For inspection if needed
    }

# --- Test Cases ---

def test_dry_run_mode(setup_test_environment):
    """
    Tests that in dry-run mode, no folders are actually deleted,
    but the counts reflect what would happen.
    """
    env = setup_test_environment
    
    # Call the function directly with explicit arguments
    deleted_count, not_found_count = remove_checkpoints.delete_checkpoints_folders(env["job_file_path"], dry_run=True)

    # Assert that the checkpoint folders still exist after a dry run
    assert os.path.isdir(env["checkpoints_64_path"])
    assert os.path.isdir(env["checkpoints_128_path"])
    assert os.path.isdir(env["wrong_checkpoints_path"]) # This one should never be touched

    # Assert counts: 0 deleted in dry run, 4 not found (3 distinct ones, 1 from empty line)
    assert deleted_count == 0
    # Expected not_found_count:
    # 1. /a/different/path/job_dir/script.sh (parent /a/different/path/job_dir has no checkpoints)
    # 2. /another/job/path/script.sh (parent /another/job/path has no checkpoints, the `checkpoints` is *under* it)
    # 3. /non/existent/base/dir/job_folder/test.sh (parent /non/existent/base/dir/job_folder does not exist)
    assert not_found_count == 3


def test_real_deletion_mode(setup_test_environment):
    """
    Tests that in real deletion mode, the correct folders are deleted.
    """
    env = setup_test_environment
    print(env["job_file_path"])
    with open(env["job_file_path"], 'r') as fd:
        print(fd.readlines())
    # Call the function directly with explicit arguments
    deleted_count, not_found_count = remove_checkpoints.delete_checkpoints_folders(
        env["job_file_path"], dry_run=False)
    print(f'deleted count {deleted_count}, not_found count {not_found_count}')
    # Assert that the correct checkpoint folders are deleted
    assert not os.path.exists(env["checkpoints_64_path"])
    assert not os.path.exists(env["checkpoints_128_path"])

    # Assert that the 'wrong' checkpoints folder is NOT deleted
    assert os.path.isdir(env["wrong_checkpoints_path"])

    # Assert counts: 2 deleted, 4 not found
    assert deleted_count == 4
    assert not_found_count == 3


def test_job_file_not_found(tmp_path, capsys):
    """
    Tests handling of a non-existent job file.
    `capsys` is a pytest fixture to capture stdout/stderr.
    """
    non_existent_job_file = str(tmp_path / "non_existent_jobs.txt")
    
    # Call the function directly with explicit arguments
    deleted_count, not_found_count = remove_checkpoints.delete_checkpoints_folders(
        non_existent_job_file, dry_run=True)
    
    # The function should return 0, 0 on an error for missing file
    assert deleted_count == 0
    assert not_found_count == 0
    
    captured = capsys.readouterr()
    assert f"Error: Job file '{non_existent_job_file}' not found." in captured.err # Check stderr