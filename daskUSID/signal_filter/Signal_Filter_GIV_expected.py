#!/usr/bin/env python
# coding: utf-8

# ***Import necessary packages.***
# While testing dask_process() class, make sure to call on functions from that class, not the regular process() class.

# In[3]:


import pyUSID as usid
from pyUSID.processing.comp_utils import parallel_compute, get_MPI, group_ranks_by_socket, get_available_memory
from dask.distributed import Client
import dask.array as da


# In[4]:


import os
os.listdir()


# In[5]:


import h5py
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
import tempfile

# I am importing pycroscopy from the original package but you will have to change yours
import sys
#sys.path.append('/Users/syz/PycharmProjects/pyUSID/')
sys.path.append('/Users/syz/PycharmProjects/pycroscopy/')
#from pycroscopy.processing import SignalFilter
#from pycroscopy.processing import fft
# You will need to do something like:
from fft import LowPassFilter
from dask_signal_filter import SignalFilter


# ***Creates temporary file*** as to not mess up original file.

# In[6]:


orig_path = 'pzt_nanocap_6_just_translation_copy.h5'

with tempfile.TemporaryDirectory() as tmp_dir:
    h5_path = tmp_dir + 'gline.h5'
    copyfile(orig_path, h5_path)


# ***Finds and assigns main dataset, Raw_Data, to h5_main***

# In[7]:


h5_f = h5py.File(h5_path, mode='r+')
# Not necessary I think but Chris used it
h5_f.atomic = True # This doesn't seem to make any difference

h5_grp = h5_f['Measurement_000/Channel_000']
h5_main = h5_grp['Raw_Data']
h5_main = usid.USIDataset(h5_main)


# 

# In[8]:


samp_rate = h5_grp.attrs['IO_samp_rate_[Hz]']
num_spectral_pts = h5_main.shape[1]

frequency_filters = [LowPassFilter(num_spectral_pts, samp_rate, 10E+3)]
noise_tol = 1E-6

sig_filt = SignalFilter(h5_main, frequency_filters=frequency_filters,
                           noise_threshold=noise_tol, write_filtered=True,
                           write_condensed=False, num_pix=1, 
                        verbose=True)

# Set verbose=True to get more information on your print statements


# ***Calls compute function.***

# In[11]:


client = Client()
sig_filt._read_data_chunk()
dat = sig_filt.data
a = client.submit(sig_filt.test, pix_ind=0)


# In[ ]:


x = da.random.random((10000, 10000), chunks=(1000, 1000))
def inc(x):
    return x + 1
L = client.map(inc, dat)
L
#h5_filt_grp = sig_filt.compute(override=True)


# ***Preparing for visualization.***

# In[ ]:


bias_vec = 6 * np.sin( np.linspace(0, 2*np.pi, 500))
h5_filt = h5_filt_grp['Filtered_Data']
row_ind = 40
filt_row = h5_filt[row_ind].reshape(-1, bias_vec.size)
raw_row = h5_main[row_ind].reshape(-1, bias_vec.size)


# ***Visualization.***

# In[ ]:


plots_on_side = 3

fig, axes = plt.subplots(nrows=plots_on_side, ncols=plots_on_side, figsize=(15, 15))
for axis, col_ind in zip(axes.flat, np.linspace(0, filt_row.shape[0]-1, plots_on_side ** 2, dtype=np.uint8)):
    axis.plot(bias_vec, raw_row[col_ind], 'r')
    axis.plot(bias_vec, filt_row[col_ind], 'k')
    axis.set_title('Row {} Col {}'.format(row_ind, col_ind))
axis.legend(['Raw', 'Filtered'])
fig.tight_layout()


# In[ ]:


h5_f.close()

