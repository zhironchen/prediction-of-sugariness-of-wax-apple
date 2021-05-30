import _pickle as pkl
import os
import numpy as np
import h5py
import pdb
# this script is created for transfer .mat file into pkl file, thus our original HSIs are .mat file 
type2_mat_dir = 'location/to/save/your/data/'
for i, name in enumerate(type2_mat_dir):
    location = 'dir/to/save/pkl/data'
    bands_idx = []
    arrays = []
    f = h5py.File(type2_mat_dir+name, 'r')
    wavelength_10 = np.array(f['DATA']['Wavelength_Idx'])
    data = np.array(f['DATA']['Fullband'][:, :, :])
    print(name)
    ref = f['DATA']['Brix'][0]
    brix = float(''.join(chr(i) for i in f[ref[0]]))
    name = name[:-4]
    with open(type2_mat_dir+"data/"+name+".pkl", "wb") as f1:
        #data format : (width, length, channels(bands))       
        pkl.dump(np.array(data).reshape(data.shape[2], data.shape[1], data.shape[0]), f1)
        print("data:"+name)
    with open(type2_mat_dir+"brix/"+name+".pkl", "wb") as f2:
        pkl.dump(np.array(brix), f2)
        print("brix:"+name)
    with open(type2_mat_dir+"wavelength/"+name+".pkl", "wb") as f3:
        pkl.dump(np.array(wavelength_10), f3)
        print("wavelength:"+name)