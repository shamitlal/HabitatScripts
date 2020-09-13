import pickle 
import sys 
import matplotlib.pyplot as plt 
import os 
import ipdb 
st = ipdb.set_trace

basepath = "/hdd/habitat_scenes_data_automated"
datapath = os.path.join(basepath, sys.argv[1])
for picklefile in os.listdir(datapath):
    picklefile = os.path.join(datapath, picklefile)
    p = pickle.load(open(picklefile, 'rb'))
    rgb = p['rgb_camX']
    plt.imshow(rgb[:,:,:3])
    plt.show(block=True)

