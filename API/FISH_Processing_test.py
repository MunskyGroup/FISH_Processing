from unittest import TestCase
from src.fish_analyses import *
#from skimage.io import imread
import cv2
class TestCellSegementation(TestCase):
    def setUp(self) -> None:
        self.object=CellSegmentation
    def test0(self):
        image=imread("/Users/mpmay/Projects/APS2023/LearnFastApi/FISH_Processing/dataBases/test_data/ROI002_XY1620755646_Z00_T0_merged.tif")
        image=image[10,:,:,:]
        origionalDimensions=image.shape
        image=cv2.resize(image, dsize=(64,64), interpolation=cv2.INTER_CUBIC)
        print(image)

        self.object=CellSegmentation(image)
        self.object.channels_with_nucleus=0
        self.object.show_plots=False
        self.object.diameter_nucleus=15
        [cell_masks,nuc_masks,difference]=self.object.calculate_masks()
        nuc_masks=cv2.resize(nuc_masks,dsize=origionalDimensions[0:2], interpolation=cv2.INTER_CUBIC)
        print("Number of nuc mask cells found : {0}".format(np.max(nuc_masks)))


class TestCellpose(TestCase):
    def setUp(self) -> None:
        self.object=Cellpose

    def test0(self):
        image = imread("/Users/mpmay/Projects/APS2023/LearnFastApi/FISH_Processing/dataBases/test_data/ROI002_XY1620755646_Z00_T0_merged.tif")
        image = image[10, :, :, :]
        origionalDimensions = image.shape
        image = cv2.resize(image, dsize=(64,64), interpolation=cv2.INTER_CUBIC)
        print(image)
        self.object = Cellpose(image)
        self.object.channels = [0,0]
        self.object.diameter = 15
        masks= self.object.calculate_masks()
        masks = cv2.resize(masks, dsize=origionalDimensions[0:2], interpolation=cv2.INTER_CUBIC)
        print("Number of cells found : {0}".format(np.max(masks)))

class TestCellposeDropdown(TestCase):
    def setUp(self) -> None:
        self.object=Cellpose

    def test0(self):
        image = imread("/Users/mpmay/Projects/APS2023/LearnFastApi/FISH_Processing/dataBases/test_data/ROI002_XY1620755646_Z00_T0_merged.tif")
        image = image[10, :, :, :]
        origionalDimensions = image.shape
        print(image)
        self.object = Cellpose(image)
        self.object.channels = [0,0]
        self.object.diameter = 15
        masks= self.object.calculate_masks()
        print("Number of cells found : {0}".format(np.max(masks)))


class TestSpotDetection(TestCase):
    def setUp(self) -> None:
        self.object=SpotDetection
    def domain(self):
        self.object=SpotDetection()
        self.object.cellpose=Cellpose()
        self.object.cellpose.diameter=5
    def test0(self):
        image = imread(
            "/Users/mpmay/Projects/APS2023/LearnFastApi/FISH_Processing/dataBases/test_data/ROI002_XY1620755646_Z00_T0_merged.tif")
        image = image[10, :, :, :]
        self.object=SpotDetection(image,masks)
        self.object.channels_with_FISH=1
        self.object.get_dataframe()


class TestPipeline(TestCase):
    def setUp(self) -> None:
        image = imread(
            "/Users/mpmay/Projects/APS2023/LearnFastApi/FISH_Processing/dataBases/test_data/ROI002_XY1620755646_Z00_T0_merged.tif")
        image = image[10, :, :, :]
        image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        self.object=PipelineFISH(image)
        self.object.save_all_images=False
        self.object.show_plots=False
        self.object.diameter_nucleus=15
        self.object.

class SpotCounter(TestCase):
    def setUp(self) -> None:
        self.object=SpotDetection()
    def test1(self):
        self.object.cellpose=Cellpose()
        self.object.spotcount(images)
        self.object.cellpose=CellposeFast()
        self.object.spotCount()

class Cellpose:
    def detect(self,image):
        #todo
        pass

class CellposeFast:
    def detect(self,image):
        #detect very fast
        pass