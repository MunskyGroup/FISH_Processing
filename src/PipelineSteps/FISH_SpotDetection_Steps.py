from ufish.api import UFish
import matplotlib.pyplot as plt
from pycromanager import Dataset
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src import PipelineStepsClass, StepOutputsClass, SingleStepCompiler

class UFISH_SpotDetection_Step(PipelineStepsClass):
    def __init__(self):
        super().__init__()


    def main(self, image, FISHChannel, display_plots:bool = False, **kwargs):
        rna = image[:, :, :, FISHChannel[0]]
        rna = rna.squeeze()

        ufish = UFish()
        ufish.load_weights()

        pred_spots, enh_img = ufish.predict(rna)

        print(pred_spots)

        if display_plots:
            ufish.plot_result(np.max(rna, axis=0) if len(rna.shape) == 3 else rna, pred_spots)
            plt.show()
            plt.imshow(np.max(rna, axis=0) if len(rna.shape) == 3 else rna)
            plt.show()
            plt.imshow(np.max(enh_img, axis=0) if len(enh_img.shape) == 3 else enh_img)
            plt.show()





if __name__ == "__main__":
    ds = Dataset(r"C:\Users\Jack\Desktop\H128_Tiles_100ms_5mW_Blue_15x15_10z_05step_2")
    kwargs = {'nucChannel': [0], 'FISHChannel': [0],
          'user_select_number_of_images_to_run': 5}
    compiler = SingleStepCompiler(ds, kwargs)
    output = compiler.sudo_run_step(UFISH_SpotDetection_Step)
    



























