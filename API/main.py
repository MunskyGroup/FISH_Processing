# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


from fastapi import FastAPI
from pydantic import BaseModel
#import cv2
from skimage.transform import resize
from FISH_Processing.src.fish_analyses import *
app = FastAPI()

class AquisitionScheduleRequest(BaseModel):
    name:str
    acq_script:str

class ImageProcessRequest(BaseModel):
    image:str
    sizeX:int
    sizey:int
    image_process:str='cell_detection'

@app.get("/test_cellmask")
async def test_cellmask():
    image = imread( "/Users/mpmay/Projects/APS2023/LearnFastApi/FISH_Processing/dataBases/test_data/ROI002_XY1620755646_Z00_T0_merged.tif")
    image = image[10, :, :, :]
    origionalDimensions = image.shape
    #image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    image = resize(image, (64, 64), anti_aliasing=True)
    cellSeg = Cellpose(image)
    cellSeg.channels = [0, 0]
    cellSeg.diameter = 15
    masks = cellSeg.calculate_masks()
    masks = cv2.resize(masks, dsize=origionalDimensions[0:2], interpolation=cv2.INTER_CUBIC)
    return "Completed Test"

@app.get("/cell_mask_from_file")
async def cell_mask_from_file(filename:str):
    print(filename)
    image = imread(filename)
    image = image[10, :, :, :]
    origionalDimensions = image.shape
    #image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    image = resize(image, (64, 64), anti_aliasing=True)
    cellSeg = Cellpose(image)
    cellSeg.channels = [0, 0]
    cellSeg.diameter = 15
    masks = cellSeg.calculate_masks()
    masks = cv2.resize(masks, dsize=origionalDimensions[0:2], interpolation=cv2.INTER_CUBIC)
    return "Completed Test"

@app.get("spot_count_from_file")
async def cell_mask_from_file(filename:str):
    pass