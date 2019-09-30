import h5py
from PIL import Image

fileName = 'data.h5'
numOfSamples = 10000
with h5py.File(fileName, "w") as out:
  out.create_dataset("X_train",(numOfSamples,256,256,3),dtype='u1')
