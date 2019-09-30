import h5py
from PIL import Image

fileName = 'data.h5'
numOfSamples = 13233
with h5py.File(fileName, "w") as out:
    out.create_dataset("X",(numOfSamples,[:100] ,256,3),dtype='u1')
