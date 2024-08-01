import pathlib
import re
from os import listdir
from os.path import isfile, join

from skimage.io import imread


class ReadImages():
    '''
    This class reads all tif images in a given folder and returns each image as a Numpy array inside a list, the names of these files, path, and the number of files.
    
    Parameters
    
    directory: str or PosixPath
        Directory containing the images to read.
    '''    
    def __init__(self, directory:str,number_of_images_to_process=None):
        if type(directory)== pathlib.PosixPath or type(directory)== pathlib.WindowsPath:
            self.directory = directory
        else:
            self.directory = pathlib.Path(directory)
        self.number_of_images_to_process = number_of_images_to_process
    def read(self):
        '''
        Method takes all the images in the folder and merges those with similar names.
        
        Returns
        
        list_images : List of NumPy arrays 
            List of NumPy arrays with format np.uint16 and dimensions [Z, Y, X, C] or [T, Y, X, C]. 
        path_files : List of strings 
            List of strings containing the path to each image.
        list_files_names : List of strings 
            List of strings where each element is the name of the files in the directory.
        number_files : int 
            The number of images in the folder.
        '''
        list_files_names_complete = sorted([f for f in listdir(self.directory) if isfile(join(self.directory, f)) and ('.tif') in f], key=str.lower)  # reading all tif files in the folder
        list_files_names_complete.sort(key=lambda f: int(re.sub('\D', '', f)))  # sorting the index in numerical order
        path_files_complete = [ str(self.directory.joinpath(f).resolve()) for f in list_files_names_complete ] # creating the complete path for each file
        number_files = len(path_files_complete)
        list_images_complete = [imread(str(f)) for f in path_files_complete]
        
        # This section reads the images to process in the repository.  If the user defines "number_of_images_to_process" as an integer N the code will process a subset N of these images. 
        if (self.number_of_images_to_process is None) or (self.number_of_images_to_process>number_files):
            list_images =list_images_complete
            path_files = path_files_complete
            list_files_names = list_files_names_complete
        else:
            list_images = list_images_complete[0:self.number_of_images_to_process]
            path_files = path_files_complete[0:self.number_of_images_to_process]
            list_files_names = list_files_names_complete[0:self.number_of_images_to_process]
            number_files = self.number_of_images_to_process
        return list_images, path_files, list_files_names, number_files
