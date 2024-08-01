# from PipelineSettings import PipelineSettings
# from ExperimentClass import Experiment
# from MicroscopeClass import ScopeClass
# from PipelineDataClass import PipelineDataClass
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

    def load_in_attributes(self):
        pass

    def main(self, ):
        pass

    def run(self, pipelineData:PipelineDataClass = None, pipelineSettings:PipelineSettings = None,
            terminatorScope:ScopeClass = None, experiment:Experiment = None):
        self.pipelineData = pipelineData
        self.pipelineSettings = pipelineSettings
        self.terminatorScope = terminatorScope
        self.experiment = experiment
        self.load_in_attributes()
        self.check_setting_requirements()
        return self.main()


class PipelineStepsClass(StepClass):
    def __init__(self):
        self.freeze = False
        self.is_first_run = True

    def run(self, id:int = None, pipelineData: PipelineDataClass = None, pipelineSettings: PipelineSettings = None,
            terminatorScope: ScopeClass = None, experiment: Experiment = None):
        self.pipelineData = pipelineData
        self.pipelineSettings = pipelineSettings
        self.terminatorScope = terminatorScope
        self.experiment = experiment
        if id is None:  # allows for pipelineSteps to be run a pre or postPipeline
            for img_index in range(min(self.pipelineSettings.user_select_number_of_images_to_run,
                                       self.experiment.number_of_images_to_process)):
                self.load_in_attributes(img_index)
                self.on_first_run(img_index)
                single_step_output = self.main(id=img_index)
                if img_index == 0:
                    output = single_step_output
                else:
                    output.append(single_step_output)
            return output
        else:
            self.load_in_attributes(id)
            self.on_first_run(id)
            return self.main(id=id)

    def main(self, id):
        pass

    def load_in_attributes(self, id):
        pass

    def on_first_run(self, id):
        if self.is_first_run:
            self.first_run(id)
            self.is_first_run = False
            return True
        else:
            return False
    
    def first_run(self, id:int):
        pass
        

class postPipelineStepsClass(StepClass):
    def __init__(self):
        super().__init__()


class prePipelineStepsClass(StepClass):
    def __init__(self):
        super().__init__()
        self.ModifyPipelineData = False
