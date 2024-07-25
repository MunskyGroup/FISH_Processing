# Many of the output classes will be the same, so we can create a base class and then inherit from it
# however, they will have differences on if they modify a PipelineDataClass or if they are the final output

# The Step Classes will be similar however they will have differences on the inputs they act on 
class OutputClass:
    def __init__(self):
        pass

    def append(self):
        pass


class StepOutputsClass(OutputClass):
    # this will be the final output of the pipeline, this will be the final output of the pipeline

    def __init__(self):
        super().__init__()

    def append(self, newOutputs):
        pass


class PipelineOutputsClass(OutputClass):
    # in pipelineData we will store many of these, these will consist of the direct results from the steps.
    # This could const of dataframes, images, etc.
    # the append function will be used to add the output to add the outputs from each image to itself.
    def __init__(self):
        super().__init__()

    def append(self, output:StepOutputsClass):
        if hasattr(self, output.__class__.__name__):
            getattr(self, output.__class__.__name__).append(output)
        else:
            setattr(self, output.__class__.__name__, output)


class PrePipelineOutputsClass(OutputClass):
    # this will be the final output of the pipeline, this will be the final output of the pipeline
    def __init__(self):
        pass

    def append(self, newOutputs):
        setattr(self, newOutputs.__class__.__name__, newOutputs)