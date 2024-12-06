import matplotlib.pyplot as plt
import numpy as np
import h5py
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from src.Parameters import Parameters



class Display:
    def displayImage(self, position: int = 0, timepoint: int = 0, channel: int = 0, zslice: int = 0):
        params = Parameters.get_parameters()
        images = params['images']
        image = images[position, timepoint, channel, zslice]
        plt.imshow(image)
        # turn of axis
        plt.axis('off')
        plt.show()
        return image

    def displayMask(self, position: int = 0, timepoint: int = 0, channel: int = 0, zslice: int = 0, label: int = None):
        params = Parameters.get_parameters()
        masks = params['masks']
        mask = masks[position, timepoint, channel, zslice]
        if label is not None:
            mask[mask != label] = 0
        plt.imshow(mask)
        plt.axis('off')
        plt.show()
        return mask

    def displayImage_maxProject(self, position: int = 0, timepoint: int = 0, channel: int = 0):
        # TODO: time these parts
        params = Parameters.get_parameters()
        images = params['images']
        image = np.max(images[position, timepoint, channel, :, :, :].compute(), axis=0)
        
        plt.imshow(image)
        plt.clim(0, np.percentile(image.flatten(), 99.99))
        plt.axis('off')
        plt.show()
        return image

    def displayGrid_Images(self, positions: list[int] = [0, 1, 2]):
        params = Parameters.get_parameters()
        images = params['images']
        fig, axs = plt.subplots(len(positions), images.shape[2], figsize=(20, 20))
        for i, position in enumerate(positions):
            for j in range(images.shape[2]):
                axs[i, j].imshow(images[position, 0, j].max(axis=0))
        plt.axis('off')
        plt.show()

    def displayGrid_Masks(self, positions: list[int] = [0, 1, 2]):
        params = Parameters.get_parameters()
        masks = params['masks']
        fig, axs = plt.subplots(len(positions), masks.shape[2], figsize=(20, 20))
        for i, position in enumerate(positions):
            for j in range(masks.shape[2]):
                axs[i, j].imshow(masks[position, 0, j, 0])
        plt.show()

    def save_all_figures(self):
        fig_nums = plt.get_fignums()
        with h5py.File('figures.h5', 'w') as f:
            for fig_num in fig_nums:
                fig = plt.figure(fig_num)
                canvas = FigureCanvas(fig)
                canvas.draw()
                img_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
                img_array = img_array.reshape(fig.bbox.height, fig.bbox.width, 3)
                f.create_dataset(f"figure_{fig_num}", data=img_array)
                plt.close(fig_num)
    
    def display_all_saved_figures(self):
        # Now to display the saved figures # TODO: I dont know how this will save so fix it afterwards
        # with h5py.File('figures.h5', 'r') as f:
        #     for fig_num in fig_nums:
        #         # Get the saved image from the file
        #         img_array = f[f"figure_{fig_num}"][:]
                
        #         # Display the image using plt.imshow
        #         plt.figure(fig_num)
        #         plt.imshow(img_array)
        #         plt.axis('off')  # Turn off axes
        #         plt.show()
        pass
