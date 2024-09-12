from pycromanager import Dataset
import inspect
from .ExperimentClass import Experiment
from .PipelineSettings import PipelineSettings
from .MicroscopeClass import ScopeClass
import numpy as np





class SingleStepCompiler:
    def __init__(self, dataset: Dataset, kwargs: dict = {}):
        self.kwargs = kwargs

        # convert dataset to list of images
        self.list_images, self.map_id_imgprops, zstep = self.convert_dataset_to_zxyc(dataset)
        if np.isclose(zstep, 0):
            zstep = 500   #  TODO: This might be a shity assumption but I am not sure how much depth a single slice is 
        else:
            zstep = zstep * 1000 # convert um to nm



        # compile into kwargs
        self.kwargs['list_images'] = self.list_images
        self.kwargs['map_id_imgprops'] = self.map_id_imgprops
        self.kwargs['verbose'] = True
        self.kwargs['display_plots'] = True
        self.kwargs['voxel_size_z'] = zstep
        kwargs = self.kwargs
        default_settings = {**Experiment(**kwargs).__dict__, **PipelineSettings(**kwargs).__dict__, **ScopeClass(**kwargs).__dict__}
        for key, value in default_settings.items():
            if key not in self.kwargs.keys():
                self.kwargs[key] = value
    

    def sudo_run_step(self, function):
        function = function()
        signature = inspect.signature(function.main)
        kwargs = self.kwargs
        overall_output = None
        num_cells_ran = 0
        if any(k in signature.parameters.keys() for k in ['id', 'image']):
            for id, image in enumerate(self.list_images):
                kwargs = self.kwargs
                kwargs['id'] = id
                kwargs['image'] = image
                kwargs['image_name'] = None
                output = function.main(**kwargs)
                num_cells_ran += 1
                if overall_output is None:
                    overall_output = output
                else:
                    overall_output.append(output)
                if num_cells_ran >= self.kwargs['user_select_number_of_images_to_run']:  # not the biggest fan of this but its cheap and easy
                    break
        else:
            overall_output = function.main(**kwargs)
        self.kwargs = {**self.kwargs, **overall_output.__dict__}
        return overall_output
    
    def convert_dataset_to_zxyc(self, dataset):
        number_z_slices = max(dataset.axes['z']) + 1
        number_color_channels = max(dataset.axes['channel']) + 1
        number_of_fov = max(dataset.axes['position']) + 1
        number_of_tp = max(dataset.axes['time']) + 1

        counter = 0
        list_images_standard_format = []
        list_tps = []
        list_zs = []
        map_id_imgprops = {}
        number_of_imgs = number_of_fov * number_of_tp
        number_of_X = None
        number_of_Y = None
        zstep_list = []
        for tp in range(number_of_tp):
            z_tp = []
            for fov in range(number_of_fov):
                z_fov =[]
                if number_of_X is not None:
                    temp_image = np.zeros((number_z_slices, number_of_Y, number_of_X, number_color_channels)) 
                for z in range(number_z_slices):
                    intended_z = dataset.read_metadata(position=fov, time=tp, z=z, channel=0)['ZPosition_um_Intended']
                    z_fov.append(intended_z)
                    for c in range(number_color_channels):
                        if number_of_X is None:
                            temp_image = dataset.read_image(position=fov, time=tp, z=z, channel=c)
                            number_of_X = temp_image.shape[1]
                            number_of_Y = temp_image.shape[0]
                            temp_image = np.zeros(
                                (number_z_slices, number_of_Y, number_of_X, number_color_channels))
                        temp_image[z, :, :, c] = dataset.read_image(position=fov, time=tp, z=z, channel=c)
                        list_tps.append(tp)
                        list_zs.append(z)
                dz = [np.abs(z_fov[i] - z_fov[i + 1]) for i in range(len(z_fov) - 1)] if len(z_fov) > 1 else 0
                z_tp.append((np.mean(dz)))
                list_images_standard_format.append(temp_image)
                map_id_imgprops[counter] = {'fov_num': fov, 'tp_num': tp}
            zstep_list.append(np.mean(z_tp))
            counter += 1
        zstep = np.mean(zstep_list)

        return list_images_standard_format, map_id_imgprops, zstep
