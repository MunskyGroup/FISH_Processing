import os
# from . import Settings, Experiment, ScopeClass, DataContainer


class StepClass:
    def __init__(self):
        self.freeze = False
        self.data = None
        self.settings = None
        self.scope = None
        self.experiment = None
        self.step_output_dir = None

    def check_setting_requirements(self):
        pass

    def __str__(self):
        return self.__class__.__name__

    def load_in_attributes(self, id: int = None):
        """
        This is where the magic happens. This function will load in all the attributes of the class and return them as a dictionary.
        This allows all the bs that I decided to force on my code to not matter, and user can just write whatever they want in the main functions
        As long as the attributes are unique and saved using a step output class, this function will load them in.
        """

        kwargs_data = self.data.__dict__
        kwargs_experiment = self.experiment.__dict__
        kwargs_scope = self.scope.__dict__
        kwargs_settings = self.settings.__dict__

        for key in kwargs_data:
            try:
                step_dict = getattr(self.data, key).__dict__
                kwargs_data = {**kwargs_data, **step_dict}
                kwargs_data.pop(key)
            except AttributeError:
                pass
        
        kwargs_IDspecific = {'id': id}
        if id is not None:
            kwargs_IDspecific['image'] = self.data.list_images[id]
            kwargs_IDspecific['image_name'] = os.path.splitext(self.data.list_image_names[id])[0]
            try:
                kwargs_IDspecific['cell_mask'] = self.data.masks_complete_cells[id]
            except AttributeError:
                kwargs_IDspecific['cell_mask'] = None
            try:
                kwargs_IDspecific['nuc_mask'] = self.data.masks_nuclei[id]
            except AttributeError:
                kwargs_IDspecific['nuc_mask'] = None
            try:
                kwargs_IDspecific['cyto_mask'] = self.data.masks_cytosol[id]
            except AttributeError:
                kwargs_IDspecific['cyto_mask'] = None
        
        kwargs = {**kwargs_data, **kwargs_experiment, **kwargs_scope, **kwargs_settings, **kwargs_IDspecific}

        return kwargs
    
    def create_step_output_dir(self, output_location = None, **kwargs):
        if output_location is not None:
            self.step_output_dir = os.path.join(output_location, self.__class__.__name__)
            os.makedirs(self.step_output_dir, exist_ok=True)
        else:
            self.step_output_dir = None

    def main(self):
        pass

    def run(self, data,
            settings,
            scope,
            experiment):
        self.data = data
        self.settings = settings
        self.scope = scope
        self.experiment = experiment
        kwargs = self.load_in_attributes()
        self.create_step_output_dir(**kwargs)
        self.check_setting_requirements()
        return self.main(**kwargs)
    


class SequentialStepsClass(StepClass):
    def __init__(self):
        super().__init__()
        self.freeze = False
        self.is_first_run = True

    def run(self, id: int = None, data = None, settings = None,
            scope = None, experiment = None):
        self.data = data
        self.settings = settings
        self.scope = scope
        self.experiment = experiment

        if id is None:  # allows for pipelineSteps to be run a pre or postPipeline
            for img_index in range(min(self.settings.user_select_number_of_images_to_run,
                                       self.experiment.number_of_images_to_process)):
                kwargs = self.load_in_attributes(img_index)
                self.create_step_output_dir(**kwargs)
                self.on_first_run(img_index)
                single_step_output = self.main(**kwargs)
                if img_index == 0:
                    output = single_step_output
                else:
                    output.append(single_step_output)
            return output
        else:
            kwargs = self.load_in_attributes(id)
            self.create_step_output_dir(**kwargs)
            self.on_first_run(id)
            # print(kwargs)
            return self.main(**kwargs)

    def main(self, **kwargs):
        pass

    def on_first_run(self, id: int):
        if self.is_first_run:
            self.first_run(id)
            self.is_first_run = False
            return True
        else:
            return False
    
    def first_run(self, id: int):
        pass
        

class finalizingStepClass(StepClass):
    def __init__(self):
        super().__init__()


class IndependentStepClass(StepClass):
    def __init__(self):
        super().__init__()
        self.ModifyPipelineData = False
