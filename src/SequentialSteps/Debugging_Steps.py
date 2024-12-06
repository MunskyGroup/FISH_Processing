import matplotlib.pyplot as plt
import os
import sys
import numpy as np



from src.GeneralStep import SequentialStepsClass



class DisplaySequentialParams(SequentialStepsClass):
    def __init__(self):
        super().__init__()

    def main(self, **kwargs):
        plt.imshow(np.max(kwargs['nuc_mask'], axis=0))
        plt.show()
        plt.imshow(np.max(kwargs['cell_mask'], axis=0))
        plt.show()

        for k, v in kwargs.items():
            print(f'{k}: {v}')












