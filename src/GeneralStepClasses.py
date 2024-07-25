# from PipelineSettings import PipelineSettings
# from ExperimentClass import Experiment
# from MicroscopeClass import ScopeClass
# from PipelineDataClass import PipelineDataClass
from . import PipelineSettings, Experiment, ScopeClass, PipelineDataClass

class StepClass():
    def __init__(self):
        self.freeze = False

    def __str__(self):
        return self.__class__.__name__

    def check_setting_requirements(self):
        pass

    def main(self):
        pass

    def run(self, pipelineData:PipelineDataClass, pipelineSettings:PipelineSettings, terminatorScope:ScopeClass, experiment:Experiment):
        return self.main(pipelineData=pipelineData, pipelineSettings=pipelineSettings, terminatorScope=terminatorScope, experiment=experiment)

class PipelineStepsClass(StepClass):
    def __init__(self):
        self.freeze = False
        self.is_first_run = True

    def run(self, id:int, pipelineData:PipelineDataClass, pipelineSettings:PipelineSettings, terminatorScope:ScopeClass, experiment:Experiment):
        self.on_first_run()
        return self.main(id=id, pipelineData=pipelineData, pipelineSettings=pipelineSettings, terminatorScope=terminatorScope, experiment=experiment)

    def on_first_run(self):
        if self.is_first_run:
            self.first_run()
            self.is_first_run = False
            return True
        else:
            return False
    
    def first_run(self):
        pass
        

class postPipelineStepsClass(StepClass):
    def __init__(self):
        super().__init__()


class prePipelineStepsClass(StepClass):
    def __init__(self):
        super().__init__()
        self.ModifyPipelineData = False
