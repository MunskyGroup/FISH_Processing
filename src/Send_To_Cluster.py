import paramiko
import os
import yaml
import sys
import pickle
import shutil

# Get the path of the current script (or current working directory)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Append the parent directory to sys.path
sys.path.append(parent_dir)

def run_on_cluster(remote_path: str, local_file: str, path_to_config_file: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config_cluster.yml')):
    # Load the configuration
    conf = yaml.safe_load(open(str(path_to_config_file)))
    usr = str(conf['user']['username'])
    pwd = str(conf['user']['password'])
    remote_address = str(conf['user']['remote_address'])
    port = 22


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
    sbatch_command = f'sbatch runner_pipeline.sh {remote_picklefile_path} /dev/null 2>&1 & disown'

    # Execute the command on the cluster
    # Combine commands to change directory and execute the batch script
    combined_command = f'cd {remote_path}; {sbatch_command}'

    stdin, stdout, stderr = ssh.exec_command(combined_command)
    stdout.channel.recv_exit_status()  # Wait for the command to complete

    # Print any output from the command
    print(stdout.read().decode())
    print(stderr.read().decode())

    # Close the SSH connection
    ssh.close()



if __name__ == "__main__":
    pass



    