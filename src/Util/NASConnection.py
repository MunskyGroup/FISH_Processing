import os
import pathlib
import shutil
import socket

import yaml
from smb.SMBConnection import SMBConnection


class NASConnection():
    '''
    This class is intended to establish a connection between Network-Attached storage and a remote (or local) computer using `pysmb <https://github.com/miketeo/pysmb>`_ . The class allows the user to connect to NAS, download specific files, and write backfiles to NAS.
    This class doesn't allow the user to delete, modify or overwrite files in NAS. For a complete description of pysmb check the `pysmb documentation <https://pysmb.readthedocs.io/en/latest/>`_ .
    To use this class, you need to:
    
    1) Use the university's network or use the two-factor authentication to connect to the university's VPN.
    2) You need to create a configuration YAML file with the following format:
    
    .. code-block:: bash

        user:
        username: name_of_the_user_in_the_nas_server
        password: user_password_in_the_nas_server 
        remote_address : ip or name for the nas server
        domain: domain for the nas server 
    
    Parameters
    
    path_to_config_file : str, or Pathlib object
        The path in the local computer contains the config file.
    share_name: str
        Name of the share partition to access in NAS. The default is 'share'.
    '''
    def __init__(self,path_to_config_file,share_name = 'share'):
        # Loading credentials
        conf = yaml.safe_load(open(str(path_to_config_file)))
        usr = str(conf['user']['username'])
        pwd = str(conf['user']['password'])
        remote_address = str(conf['user']['remote_address'])
        domain = str(conf['user']['domain'])
        # LOCAL NAME
        try:
            local_name = socket.gethostbyname(socket.gethostname())
        except:
            local_name = socket.gethostname()
        # SERVER NAME
        self.share_name = share_name
        self.server_name, _, _ = socket.gethostbyaddr(remote_address)
        # Defining the connection to NAS
        self.conn = SMBConnection(username=usr, password=pwd, domain=domain, my_name=local_name, remote_name=str(self.server_name), is_direct_tcp=True)
    def connect_to_server(self,timeout=60):
        '''
        This method establishes the connection to the NAS.
        
        Parameters 
        
        timeout : int, optional
            Time in seconds to maintain a connection with the NAS. The default is 60 seconds.
        '''
        is_connected = self.conn.connect(str(self.server_name),timeout=timeout)
        if is_connected == True:
            print('Connection established')
        else:
            print('Connection failed')
        return self.conn
    
    def read_files(self, remote_folder_path, timeout=60):
        '''
        This method reads all files from a NAS directory
        
        Parameters
        
        remote_folder_path : str, Pathlib obj
            The path in the remote folder to download.
        timeout : int, optional
            Time in seconds to maintain a connection with the NAS. The default is 60 seconds.
        '''
        # Connecting to NAS
        is_connected = self.conn.connect(str(self.server_name),timeout=timeout)
        if is_connected == True:
            print('Connection established')
        else:
            print('Connection failed')
        # Converting the paths to a Pathlib object
        if type(remote_folder_path)==str:
            remote_folder_path = pathlib.Path(remote_folder_path)
        # Iterate in the folder to download all tif files
        list_files =[]
        list_dir = self.conn.listPath(self.share_name, str(remote_folder_path))
        for file in list_dir:
            list_files.append(file.filename)
        return list_files

    def download_file(self, remote_file_path, local_folder_path, timeout=600):
        '''
        This method download an specific file
        
        Parameters
        
        remote_file_path : str, Pathlib obj
            The path in the remote file to download.
        local_folder_path : str, Pathlib obj
            The path in the local computer where the files will be copied.
        timeout : int, optional
            Time in seconds to maintain a connection with the NAS. The default is 60 seconds.
        '''
        # Connecting to NAS
        is_connected = self.conn.connect(str(self.server_name),timeout=timeout)
        if is_connected == True:
            print('Connection established')
        else:
            print('Connection failed')
        # Converting the paths to a Pathlib object
        if type(local_folder_path) == str:
            local_folder_path = pathlib.Path(local_folder_path)
        if type(remote_file_path)==str:
            remote_file_path = pathlib.Path(remote_file_path)
        # Making the local directory
        if not (os.path.exists(local_folder_path)) :
            os.makedirs(str(local_folder_path))
        filename = remote_file_path.name
        fileobj = open(remote_file_path.name,'wb')
        self.conn.retrieveFile(self.share_name, str(remote_file_path), fileobj)
        fileobj.close()
        # moving files in the local computer
        shutil.move(pathlib.Path().absolute().joinpath(filename), local_folder_path.joinpath(filename))
        print('Files downloaded to: ' + str(local_folder_path.joinpath(filename)))
        return None
    
    def copy_files(self, remote_folder_path, local_folder_path, timeout=600, file_extension =['.index','.tif']):
        '''
        This method downloads tif files from NAS to a temporal folder in the local computer.
        
        Parameters
        
        remote_folder_path : str, Pathlib obj
            The path in the remote folder to download.
        local_folder_path : str, Pathlib obj
            The path in the local computer where the files will be copied.
        timeout : int, optional
            Time in seconds to maintain a connection with the NAS. The default is 60 seconds.
        file_extension : str, optional.
            String representing the file type to download.
        '''
        # Connecting to NAS
        is_connected = self.conn.connect(str(self.server_name),timeout=timeout)
        if is_connected == True:
            print('Connection established')
        else:
            print('Connection failed')
        # Converting the paths to a Pathlib object
        if type(local_folder_path) == str:
            local_folder_path = pathlib.Path(local_folder_path)
        if type(remote_folder_path)==str:
            remote_folder_path = pathlib.Path(remote_folder_path)
        # Making the local directory
        if (os.path.exists(local_folder_path))  and  (str(local_folder_path.name)[0:5] ==  'temp_'):
            shutil.rmtree(local_folder_path)
        os.makedirs(str(local_folder_path))
        # Iterate in the folder to download all tif files
        list_dir = self.conn.listPath(self.share_name, str(remote_folder_path))
        for file in list_dir:
            #if (file.filename not in ['.', '..']) and (file_extension in file.filename) :
            if (file.filename not in ['.', '..']) and  any(file.filename.endswith(ext) for ext in file_extension):
                print ('File Downloaded :', file.filename)
                fileobj = open(file.filename,'wb')
                self.conn.retrieveFile(self.share_name, str( pathlib.Path(remote_folder_path).joinpath(file.filename) ),fileobj)
                fileobj.close()
                # moving files in the local computer
                shutil.move(pathlib.Path().absolute().joinpath(file.filename), local_folder_path.joinpath(file.filename))
        print('Files downloaded to: ' + str(local_folder_path))
        return None
    
    def write_files_to_NAS(self, local_file_to_send_to_NAS, remote_folder_path,  timeout=600):
        '''
        This method writes files from a local computer to NAS 
        
        Parameters
        
        local_file_to_send_to_NAS : str, Pathlib obj
            The path in the file to send to the NAS.
        remote_folder_path : str, Pathlib obj
            The path in the remote folder to download.
        timeout : int, optional
            Time in seconds to maintain a connection with the NAS. The default is 60 seconds.
        '''
        # Connecting to NAS
        is_connected = self.conn.connect(str(self.server_name),timeout=timeout)
        if is_connected == True:
            print('Connection established')
        else:
            print('Connection failed')
        # Converting the paths to a Pathlib object
        if type(local_file_to_send_to_NAS) == str:
            local_file_to_send_to_NAS = pathlib.Path(local_file_to_send_to_NAS)
        if type(remote_folder_path)==str:
            remote_folder_path = pathlib.Path(remote_folder_path)
        # checks that the file doesn't exist on NAS. If it exist it will create a new name as follows original_name__1
        # Iterate in the folder to download all tif files
        list_dir = self.conn.listPath(self.share_name, str(remote_folder_path))
        list_all_files_in_NAS = [file.filename for file in list_dir]
        if str(local_file_to_send_to_NAS.name) not in list_all_files_in_NAS:
            with open(str(local_file_to_send_to_NAS), 'rb') as file_obj:
                self.conn.storeFile(self.share_name, str( pathlib.Path(remote_folder_path).joinpath(local_file_to_send_to_NAS.name) ) ,  file_obj )
                print ('The file was uploaded to NAS in location:', str( pathlib.Path(remote_folder_path).joinpath(local_file_to_send_to_NAS.name))  )
        return None
