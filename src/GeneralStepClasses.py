import os
from . import PipelineSettings, Experiment, ScopeClass, PipelineDataClass


class StepClass:
    def __init__(self):
        self.freeze = False
        self.pipelineData = None
        self.pipelineSettings = None
        self.terminatorScope = None
        self.experiment = None

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

        kwargs_pipelineData = self.pipelineData.__dict__
        kwargs_experiment = self.experiment.__dict__
        kwargs_terminatorScope = self.terminatorScope.__dict__
        kwargs_pipelineSettings = self.pipelineSettings.__dict__

        for key in kwargs_pipelineData:
            try:
                step_dict = getattr(self.pipelineData, key).__dict__
                kwargs_pipelineData = {**kwargs_pipelineData, **step_dict}
            except AttributeError:
                pass
            
        if id is not None:
            kwargs_IDspecific = {
                'image_index' : id,
                'image' : self.pipelineData.list_images[id],
                'image_name' : os.path.splitext(self.pipelineData.list_image_names[id])[0],
                # 'cell_mask' : self.pipelineData.masks_complete_cells[id],
                # 'nuc_mask' : self.pipelineData.masks_nuclei[id],
                # 'cyto_mask' : self.pipelineData.masks_cytosol[id],
            }
            try:
                kwargs_IDspecific['cell_mask'] = self.pipelineData.masks_complete_cells[id]
            except TypeError:
                pass
            try:
                kwargs_IDspecific['nuc_mask'] = self.pipelineData.masks_nuclei[id]
            except TypeError:
                pass
            try:
                kwargs_IDspecific['cyto_mask'] = self.pipelineData.masks_cytosol[id]
            except TypeError:
                pass
        
            kwargs = {**kwargs_pipelineData, **kwargs_experiment, **kwargs_terminatorScope, **kwargs_pipelineSettings, **kwargs_IDspecific}

        else:
            kwargs = {**kwargs_pipelineData, **kwargs_experiment, **kwargs_terminatorScope, **kwargs_pipelineSettings}

        return kwargs
    
    def create_step_output_dir(self, output_location = None, **kwargs):

        if output_location is not None:
            self.step_output_dir = os.path.join(output_location, self.__class__.__name__)
            os.makedirs(self.step_output_dir, exist_ok=True)
        else:
            self.step_output_dir = None

    def main(self):
        pass

    def run(self, pipelineData: PipelineDataClass = None,
            pipelineSettings: PipelineSettings = None,
            terminatorScope: ScopeClass = None,
            experiment: Experiment = None):
        self.pipelineData = pipelineData
        self.pipelineSettings = pipelineSettings
        self.terminatorScope = terminatorScope
        self.experiment = experiment
        kwargs = self.load_in_attributes()
        self.create_step_output_dir(**kwargs)
        self.check_setting_requirements()
        return self.main(**kwargs)
    


class PipelineStepsClass(StepClass):
    def __init__(self):
        self.freeze = False
        self.is_first_run = True

    def run(self, id: int = None, pipelineData: PipelineDataClass = None, pipelineSettings: PipelineSettings = None,
            terminatorScope: ScopeClass = None, experiment: Experiment = None):
        self.pipelineData = pipelineData
        self.pipelineSettings = pipelineSettings
        self.terminatorScope = terminatorScope
        self.experiment = experiment


        if id is None:  # allows for pipelineSteps to be run a pre or postPipeline
            for img_index in range(min(self.pipelineSettings.user_select_number_of_images_to_run,
                                       self.experiment.number_of_images_to_process)):
                kwargs = self.load_in_attributes()
                self.on_first_run(img_index)
                single_step_output = self.main(id=img_index, **kwargs)
                if img_index == 0:
                    output = single_step_output
                else:
                    output.append(single_step_output)
            return output
        else:
            kwargs = self.load_in_attributes(id)
            self.create_step_output_dir(**kwargs)
            self.on_first_run(id)
            return self.main(id=id, **kwargs)

    def main(self, id, **kwargs):
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
        

class postPipelineStepsClass(StepClass):
    def __init__(self):
        super().__init__()


class prePipelineStepsClass(StepClass):
    def __init__(self):
        super().__init__()
        self.ModifyPipelineData = False
