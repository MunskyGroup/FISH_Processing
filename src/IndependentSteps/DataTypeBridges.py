import pathlib
import pycromanager as pycro
import shutil
import tifffile
import numpy as np
import os
import sys
import inspect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src import IndependentStepClass
from src.Util import Utilities, NASConnection


#%% Useful Functions
def get_first_executing_folder():
    # Get the current stack
    stack = inspect.stack()
    
    # The first frame in the stack is the current function,
    # The second frame is the caller. However, we want the first script.
    # To find that, we look for the first frame that isn't from an internal call.
    for frame in stack:
        if frame.filename != __file__:
            # Extract the directory of the first non-internal call
            return os.path.dirname(os.path.abspath(frame.filename))

    return None




#%% Data Bridges
class Pycromanager2NativeDataType(IndependentStepClass):
    def __init__(self):
        self.experiment = None
        self.pipelineSettings = None
        self.terminatorScope = None
        self.pipelineData = None

    def run(self, data, settings, scope, experiment):

        connection_config_location = settings.connection_config_location
        self.experiment = experiment
        self.pipelineSettings = settings
        self.terminatorScope = scope
        self.pipelineData = data

        (self.local_data_dir, self.masks_dir, self.list_files_names, self.list_images_all_fov, self.list_images, \
         self.number_of_fov, self.number_color_channels, self.number_z_slices, self.number_of_timepoints,
         self.list_tps, self.list_nZ, self.number_of_imgs, self.map_id_imgprops) = self.convert_to_standard_format(
            data_folder_path=pathlib.Path(experiment.initial_data_location),
            path_to_config_file=connection_config_location,
            download_data_from_NAS=settings.download_data_from_NAS,
            use_metadata=1,
            is_format_FOV_Z_Y_X_C=1)

        # return extracted data to the experiment storage
        experiment.number_of_channels = self.number_color_channels
        experiment.number_of_timepoints = self.number_of_timepoints
        experiment.number_of_Z = self.number_z_slices
        experiment.number_of_FOVs = self.number_of_fov
        experiment.number_of_Timepoints = self.number_of_timepoints
        experiment.number_of_images_to_process = experiment.number_of_FOVs * experiment.number_of_timepoints
        experiment.list_initial_z_slices_per_image = self.list_nZ
        experiment.list_timepoints = self.list_tps
        experiment.map_id_imgprops = self.map_id_imgprops

        # PipelineData 
        data.local_data_folder = self.local_data_dir
        data.total_num_imgs = experiment.number_of_images_to_process
        data.list_image_names = self.list_files_names
        data.list_images = self.list_images
        data.num_img_2_run = min(self.pipelineSettings.user_select_number_of_images_to_run,
                                        self.experiment.number_of_images_to_process)

    def convert_to_standard_format(self, data_folder_path, path_to_config_file, download_data_from_NAS, use_metadata,
                                   is_format_FOV_Z_Y_X_C):
        path_to_masks_dir = None
        # Creating a folder to store all plots
        destination_folder = pathlib.Path().absolute().joinpath('temp_' + data_folder_path.name + '_sf')
        if pathlib.Path.exists(destination_folder):
            shutil.rmtree(str(destination_folder))
            destination_folder.mkdir(parents=True, exist_ok=True)
        else:
            destination_folder.mkdir(parents=True, exist_ok=True)

        local_data_dir, _, _, _, list_files_names_all_fov, list_images_all_fov = Utilities().read_images_from_folder(
            path_to_config_file=path_to_config_file, data_folder_path=data_folder_path, 
            path_to_masks_dir=path_to_masks_dir, download_data_from_NAS=download_data_from_NAS)
        
        if not download_data_from_NAS:
            local_data_dir = data_folder_path
        
        # Downloading data
        if use_metadata == True:
            try:
                metadata = pycro.Dataset(str(local_data_dir))
                # print(metadata.axes)
                number_z_slices = max(metadata.axes['z']) + 1
                number_color_channels = max(metadata.axes['channel']) + 1
                number_of_fov = max(metadata.axes['position']) + 1
                number_of_tp = max(metadata.axes['time']) + 1
                detected_metadata = True
                print('Number of z slices: ', str(number_z_slices), '\n',
                      'Number of color channels: ', str(number_color_channels), '\n'
                                                                                'Number of FOV: ', str(number_of_fov),
                      '\n',
                      'Number of TimePoints', str(number_of_tp), '\n', '\n', '\n')
            except:
                raise ValueError('The metadata file is not found. Please check the path to the metadata file.')
            counter = 0
            list_images_standard_format = []
            list_files_names = []
            list_tps = []
            list_zs = []
            map_id_imgprops = {}
            number_of_imgs = number_of_fov * number_of_tp
            number_of_files = len(list_files_names_all_fov)
            number_of_X = None
            number_of_Y = None
            for i in range(number_of_files):
                for tp in range(number_of_tp):
                    for fov in range(number_of_fov):
                        if not (number_of_X is None):
                            temp_image = np.zeros((number_z_slices, number_of_Y, number_of_X, number_color_channels))
                        for z in range(number_z_slices):
                            for c in range(number_color_channels):
                                if number_of_X is None:
                                    temp_image = metadata.read_image(position=fov, time=tp, z=z, channel=c)
                                    number_of_X = temp_image.shape[1]
                                    number_of_Y = temp_image.shape[0]
                                    temp_image = np.zeros(
                                        (number_z_slices, number_of_Y, number_of_X, number_color_channels))
                                temp_image[z, :, :, c] = metadata.read_image(position=fov, time=tp, z=z, channel=c)
                                list_tps.append(tp)
                                list_zs.append(z)

                        list_images_standard_format.append(temp_image)
                        list_files_names.append(
                            list_files_names_all_fov[i].split(".")[0] + '_tp_' + str(tp) + '_fov_' + str(fov) + '.tif')
                        tifffile.imsave(str(destination_folder.joinpath(list_files_names[-1])),
                                        list_images_standard_format[-1])
                        map_id_imgprops[counter] = {'fov_num': fov, 'tp_num': tp}
                        counter += 1
        masks_dir = None
        return (destination_folder, masks_dir, list_files_names, list_images_all_fov, list_images_standard_format,
                number_of_fov, number_color_channels, number_z_slices, number_of_tp, list_tps, list_zs, number_of_imgs,
                map_id_imgprops)


class FFF2NativeDataType(IndependentStepClass):
    def __init__(self):
        self.experiment = None
        self.pipelineSettings = None
        self.terminatorScope = None
        self.pipelineData = None

    def run(self, data, settings, scope, experiment):
    
            connection_config_location = settings.connection_config_location
            self.experiment = experiment
            self.pipelineSettings = settings
            self.terminatorScope = scope
            self.pipelineData = data
    
            # download the folder from NAS
            self.local_folder = 'temp_' + os.path.basename(experiment.initial_data_location)
            if not os.path.exists(self.local_folder):
                nas = NASConnection(pathlib.Path(settings.connection_config_location))
                os.makedirs(self.local_folder, exist_ok=True)   
                nas.copy_files(remote_folder_path=pathlib.Path(experiment.initial_data_location), 
                            local_folder_path=self.local_folder, 
                            file_extension=['.tif', '.zip', '.tiff', '.log'])

            # convert data to standard format
            self.convert_to_standard_format()

            # return extracted data to the experiment storage
            experiment.number_of_channels = self.number_color_channels
            experiment.number_of_timepoints = self.number_of_timepoints
            experiment.number_of_Z = self.number_z_slices
            experiment.number_of_FOVs = self.number_of_fov
            experiment.number_of_images_to_process = experiment.number_of_FOVs * experiment.number_of_timepoints
            experiment.list_timepoints = range(self.number_of_timepoints)
            experiment.map_id_imgprops = self.map_id_imgprops
    
            # PipelineData 
            data.local_data_folder = self.local_folder
            data.total_num_imgs = experiment.number_of_images_to_process
            data.list_image_names = [f.split('.')[0] for f in self.tifs]
            data.list_images = self.list_images
            data.list_nuc_masks = self.nuc_masks
            data.list_cell_masks = self.cell_masks
            data.list_cyto_masks = self.cyto_masks
            data.num_img_2_run = min(self.pipelineSettings.user_select_number_of_images_to_run,
                                            self.experiment.number_of_images_to_process)
            
    def convert_to_standard_format(self):
        self.mask_dir = None
        files = os.listdir(self.local_folder)
        self.tifs = [f for f in files if f.endswith('.tif')]
        logs = [f for f in files if f.endswith('.log')]
        mask_dirs = [f for f in files if f.startswith('masks')]

        already_made_masks = False

        unzipped_mask_dir = [f for f in mask_dirs if os.path.isdir(os.path.join(self.local_folder, f))]
        zipped_mask_dir = [f for f in mask_dirs if f.endswith('.zip')]

        mask_tifs = [f for f in mask_dirs if f.endswith('.tif')]

        if len(zipped_mask_dir) == 1 and len(mask_tifs) == 0:
            shutil.unpack_archive(os.path.join(self.local_folder, zipped_mask_dir[0]), self.local_folder)
            self.masks_dir = os.path.join(self.local_folder, zipped_mask_dir[0].split('.')[0])
            mask_tifs = [f for f in mask_dirs if f.endswith('.tif')]
            already_made_masks = True


        if len(mask_tifs) > 0:
            mask_cells = [f for f in mask_tifs if 's_cyto_R' in f]
            mask_nuclei = [f for f in mask_tifs if 'nuclei' in f]
            mask_cyto = [f for f in mask_tifs if 'cyto_no_nuclei' in f]
            already_made_masks = True
    
        # create list of images
        list_images_names = [f for f in self.tifs if not f.startswith('masks')]
        self.list_channels = [f.split('_')[-1].split('.')[0] for f in list_images_names]
        self.list_roi = [f.split('_')[0] for f in list_images_names]
        self.list_names = [f.split('_')[1] for f in list_images_names]
        self.z_slices = [f.split('_')[2] for f in list_images_names]
        self.timepoints = [f.split('_')[3] for f in list_images_names]

        self.number_of_timepoints = len(set(self.timepoints))
        # self.number_z_slices = len(set(self.z_slices))
        self.number_color_channels = len(set(self.list_channels))
        self.number_of_fov = len(set(self.list_roi))

        self.number_of_images_to_process = self.number_of_fov * self.number_of_timepoints

        self.list_images = []

        x = None
        y = None
        self.cyto_masks = []
        self.nuc_masks = []
        self.cell_masks = []
        self.map_id_imgprops = {}
        count = 0
        for t in range(self.number_of_timepoints):
            tp = list(set(self.timepoints))[t]
            for r in range(self.number_of_fov):
                fov = list(set(self.list_roi))[r]
                if x is not None:
                    temp_img = np.zeros((self.number_z_slices, y, x, self.number_color_channels))
                for c in range(self.number_color_channels):
                    channel = list(set(self.list_channels))[c]
                    search_params = [fov, channel, tp]
                    img_name = [f for f in list_images_names if all(v in f for v in search_params)][0]
                    img = tifffile.imread(os.path.join(self.local_folder, img_name))
                    if x is None:
                        y = img.shape[1]
                        z = img.shape[0]
                        x = img.shape[2]
                        self.number_z_slices = z
                        temp_img = np.zeros((self.number_z_slices, y, x, self.number_color_channels))
                    else:
                        assert x == img.shape[2]
                        assert y == img.shape[1]
                    temp_img[:, :, :, c] = img
                self.list_images.append(temp_img)
                self.map_id_imgprops[count] = {'fov_num': fov, 'tp_num': tp}
                count += 1
                search_params = [fov, tp]
                if already_made_masks:
                    cell_mask_name = [f for f in mask_cells if all(v in f for v in search_params)][0]
                    cyto_mask_name = [f for f in mask_cyto if all(v in f for v in search_params)][0]
                    nuc_mask_name = [f for f in mask_nuclei if all(v in f for v in search_params)][0]
                    self.cell_masks.append(tifffile.imread(os.path.join(self.local_folder, cell_mask_name)))
                    self.cyto_masks.append(tifffile.imread(os.path.join(self.local_folder, cyto_mask_name)))
                    self.nuc_masks.append(tifffile.imread(os.path.join(self.local_folder, nuc_mask_name)))








if __name__ == '__main__':
    from src import Experiment, Settings, ScopeClass, DataContainer
    experiment = Experiment()
    settings = Settings()
    scope = ScopeClass()
    data = DataContainer()

    experiment.initial_data_location = r'smFISH_images\Eric_smFISH_images\20230511\DUSP1_DexTimeConcSweep_10nM_75min_041223'

    FFF2NativeDataType().run(data, settings, scope, experiment)
    print(data.list_images[0].shape)
    print(len(data.list_images))
    print(len(data.list_nuc_masks))
    print(len(data.list_cell_masks))
    print(len(data.list_cyto_masks))
    print(data.list_nuc_masks[4].shape)




  