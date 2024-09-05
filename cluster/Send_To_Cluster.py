import paramiko
import os
import yaml
import sys
import pickle
from pipeline_builder import display_object_attributes
import shutil

# Get the path of the current script (or current working directory)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Append the parent directory to sys.path
sys.path.append(parent_dir)


def select_pipeline(pipeline_locations):
    Pipelines = os.listdir(pipeline_locations)

    print('Select a pipeline to run:')
    for i, pipeline in enumerate(Pipelines):
        print(f'{i}: {pipeline}')


    pipeline_index = input('Enter the index of the pipeline you want to run: ')
    while True:
        try:
            pipeline_index = int(pipeline_index)
            break
        except ValueError:
            print('Invalid input! Please enter an integer.')
            pipeline_index = input('Enter the index of the pipeline you want to run: ')

    return os.path.join(pipeline_locations, Pipelines[pipeline_index])


def update_package_for_sending(pipelines_location):
    selected_pipeline_location = select_pipeline(pipelines_location)

    # Load in dataclass
    Settings = pickle.load(open(os.path.join(selected_pipeline_location, 'settings.pkl'), 'rb'))
    Scope = pickle.load(open(os.path.join(selected_pipeline_location, 'scope.pkl'), 'rb'))
    Experiment = pickle.load(open(os.path.join(selected_pipeline_location, 'experiment.pkl'), 'rb'))
    Data = pickle.load(open(os.path.join(selected_pipeline_location, 'data.pkl'), 'rb'))

    # Update the dataclass
    Settings = display_object_attributes(Settings)
    Scope = display_object_attributes(Scope)
    Experiment = display_object_attributes(Experiment)
    Data = display_object_attributes(Data)

    # Copy the class to cluster directory
    copy_pipeline_location = os.path.join(os.getcwd(), 'cluster', os.path.basename(selected_pipeline_location))
    if os.path.exists(copy_pipeline_location):
        shutil.rmtree(copy_pipeline_location)
        shutil.copytree(selected_pipeline_location, copy_pipeline_location)

    # Rewrite the pickle files
    pickle.dump(Settings, open(os.path.join(copy_pipeline_location, 'settings.pkl'), 'wb'))
    pickle.dump(Scope, open(os.path.join(copy_pipeline_location, 'scope.pkl'), 'wb'))
    pickle.dump(Experiment, open(os.path.join(copy_pipeline_location, 'experiment.pkl'), 'wb'))
    pickle.dump(Data, open(os.path.join(copy_pipeline_location, 'data.pkl'), 'wb'))

    # Zip the package
    print( os.path.join(os.getcwd(), 'cluster'))
    shutil.make_archive(copy_pipeline_location, 'zip', os.path.join(os.getcwd(), 'cluster'), os.path.basename(copy_pipeline_location))

    return copy_pipeline_location + '.zip'


def run_on_cluster(path_to_config_file: str, local_file: str):
    # Load the configuration
    conf = yaml.safe_load(open(str(path_to_config_file)))
    usr = str(conf['user']['username'])
    pwd = str(conf['user']['password'])
    remote_address = str(conf['user']['remote_address'])
    port = 22

    remote_path = '/home/formanj/FISH_Processing/cluster'  # Path where you want to store directories_list.txt on the cluster
    remote_script = '/home/formanj/FISH_Processing/cluster/runner_cluster_TerminatorBridge.sh'

    # Create SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(remote_address, port, usr, pwd)

    # Remote path to file with all directories
    remote_picklefile_path = os.path.join(remote_path, os.path.basename(local_file))
    remote_picklefile_path = remote_picklefile_path.replace('\\', '/')

    # Transfer the file
    sftp = ssh.open_sftp()
    sftp.put(local_file, remote_picklefile_path)
    sftp.close()


    # Command to execute the batch script
    sbatch_command = f'sbatch runner_cluster_pipeline.sh {remote_picklefile_path} /dev/null 2>&1 & disown'

    # Execute the command on the cluster
    # Combine commands to change directory and execute the batch script
    combined_command = f'cd FISH_Processing/cluster; {sbatch_command}'

    stdin, stdout, stderr = ssh.exec_command(combined_command)
    stdout.channel.recv_exit_status()  # Wait for the command to complete

    # Print any output from the command
    print(stdout.read().decode())
    print(stderr.read().decode())

    # Close the SSH connection
    ssh.close()



if __name__ == "__main__":
    # Define connection parameters
    path_to_config_file = r"C:\Users\Jack\Desktop\config_keck.yml"
    pipeline_location = r'C:\Users\Jack\Documents\GitHub\FISH_Processing\Pipelines'
    # zipped_pipeline_location = update_package_for_sending(pipeline_location)
    zipped_pipeline_location = r'C:\Users\Jack\Documents\GitHub\FISH_Processing\cluster\Standard_Pipeline.zip'
    run_on_cluster(path_to_config_file, zipped_pipeline_location)



    