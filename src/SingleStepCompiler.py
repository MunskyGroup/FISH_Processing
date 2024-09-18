from pycromanager import Dataset
import inspect
from .ExperimentClass import Experiment
from .PipelineSettings import PipelineSettings
from .MicroscopeClass import ScopeClass
import numpy as np
import tkinter as tk
from tkinter import ttk

# Function to update the slider range
def update_slider_range(slider, min_entry, max_entry):
    try:
        new_min = float(min_entry.get())
        new_max = float(max_entry.get())
        if new_min < new_max:
            slider.config(from_=new_min, to=new_max)
    except ValueError:
        pass  # Ignore invalid inputs

# Function to create sliders, range inputs, and text boxes for parameters
def create_gui(parameters, runner, function):
    def run_processing():
        param_values = {param: sliders[param].get() for param in sliders}
        for key, value in param_values.items():
            if hasattr(runner.kwargs, key):
                setattr(runner.kwargs, key, value)
            else:
                runner.kwargs[key] = value
        runner.sudo_run_step(function)

    root = tk.Tk()
    root.title("Dynamic Image Processing Parameters")

    sliders = {}

    for i, param in enumerate(parameters):
        param_name, param_type, param_range = param

        frame = tk.Frame(root)
        frame.pack(pady=5)

        label = tk.Label(frame, text=param_name)
        label.pack(side=tk.LEFT)

        if param_type == 'slider':
            # Create range input fields
            range_frame = tk.Frame(frame)
            range_frame.pack(side=tk.LEFT, padx=10)

            min_label = tk.Label(range_frame, text="Min:")
            min_label.pack(side=tk.LEFT)

            min_entry = ttk.Entry(range_frame, width=5)
            min_entry.insert(0, str(param_range[0]))  # Default min value
            min_entry.pack(side=tk.LEFT)

            max_label = tk.Label(range_frame, text="Max:")
            max_label.pack(side=tk.LEFT)

            max_entry = ttk.Entry(range_frame, width=5)
            max_entry.insert(0, str(param_range[1]))  # Default max value
            max_entry.pack(side=tk.LEFT)

            # Create the slider
            slider = tk.Scale(frame, from_=param_range[0], to=param_range[1], orient=tk.HORIZONTAL)
            slider.pack(side=tk.LEFT)
            sliders[param_name] = slider

            # Update slider range when min/max values change
            update_button = tk.Button(range_frame, text="Update Range", 
                                      command=lambda s=slider, min_e=min_entry, max_e=max_entry: update_slider_range(s, min_e, max_e))
            update_button.pack(side=tk.LEFT, padx=5)

        elif param_type == 'textbox':
            entry = ttk.Entry(frame)
            entry.pack(side=tk.LEFT)
            sliders[param_name] = entry

    # Run button
    run_button = tk.Button(root, text="Run", command=run_processing)
    run_button.pack(pady=10)

    root.mainloop()


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
        # default_settings = {**Experiment(**kwargs).__dict__, **PipelineSettings(**kwargs).__dict__, **ScopeClass(**kwargs).__dict__}
        # for key, value in default_settings.items():
        #     if key not in self.kwargs.keys():
        #         self.kwargs[key] = value
    
    def sudo_run_step(self, function):
        function = function()
        signature = inspect.signature(function.main)
        kwargs = self.kwargs
        overall_output = None
        num_cells_ran = 0
        if any(k in signature.parameters.keys() for k in ['id', 'image']):
            for id, image in enumerate(self.list_images):
                print(f'========================== Running cell {id} ==========================')
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
    
    def run_sliders(self, params, function):
        create_gui(params, self, function)

