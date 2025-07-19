from WHAM import WHAM
#import json
import h5py
import numpy as np

# save single
#shot = 240730109
#wham = WHAM(shot, load_list=['bias', 'flux', 'interferometer'])
#fout = f"out/data-{shot}.h5"
#wham.save_full_h5(fout)


day = 250220000
for s in np.arange(118): #118
    shot = day + s
    wham = WHAM(shot, load_list=['interferometer'])
    #wham = WHAM(shot, load_list=['bias', 'flux'])
    # Save to a file
    fout = f"out/data-{shot}.h5"
    wham.save_data_h5(fout, level='compressed')

