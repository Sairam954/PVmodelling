import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import os
from os import listdir
from os.path import isfile, join

die_data_path = "C://Users//saira//Google Drive//ResearchPapers//Fall2020//Code//interpvfolder//FINESSE//Block2by2//"

# import glob
# print(glob.glob("/pvfigures/Q/*.png"))
#


search_dir = die_data_path
os.chdir(search_dir)
files = filter(os.path.isfile, os.listdir(search_dir))
files = [os.path.join(search_dir, f) for f in files]  # add path to each file
print(files)

files.sort(key=lambda x: os.path.getmtime(x))
dt = pd.read_csv(files[0]).to_numpy()
no_of_row = 8
no_of_columns = 10
die_x_dim = dt.shape[0]
die_y_dim = dt.shape[1]
print(dt.shape[0])
print(dt.shape[1])
wafferarray = np.zeros((dt.shape[0] * no_of_row, dt.shape[1] * no_of_columns))
print(wafferarray.shape)
no_of_dies = 80
die_number = 0
blockrowstartidx = 0
columnstartidx = 0
empty_rows = [0, 1, 6, 7]

empty_columns_for_row = {0: [0, 1, 7, 8, 9],
                         1: [0, 9],
                         6: [0, 9],
                         7: [0, 1, 7, 8, 9]}
parameter = 'FINESSE'
for i in range(0, no_of_row):
    blockrowstartidx = i * die_x_dim
    print("blockrowstartidx  :", blockrowstartidx)
    for j in range(0, no_of_columns):
        print("Die :", die_number)
        columnstartidx = j * die_y_dim
        if (i in empty_rows) and (j in empty_columns_for_row[i]):
            print("Row Range= " + str(blockrowstartidx) + ' ' + str(blockrowstartidx + die_x_dim))
            print("Column Range " + str(columnstartidx) + ' ' + str(columnstartidx + die_y_dim))
            wafferarray[blockrowstartidx:(blockrowstartidx + die_x_dim),
            columnstartidx:(columnstartidx + die_y_dim)] = 0
            die_number += 1
        else:
            print("Row Range= " + str(blockrowstartidx) + ' ' + str(blockrowstartidx + die_x_dim))
            print("Column Range " + str(columnstartidx) + ' ' + str(columnstartidx + die_y_dim))
            # print(wafferarray[blockrowstartidx:(blockrowstartidx+die_x_dim),columnstartidx:(columnstartidx+die_y_dim)].shape)
            # distribution =
            wafferarray[blockrowstartidx:(blockrowstartidx + die_x_dim),
            columnstartidx:(columnstartidx + die_y_dim)] = pd.read_csv(files[die_number]).to_numpy()
            die_number += 1

# print(pd.read_csv(str(files[die_number])).to_numpy().shape)
fig, ax = plt.subplots()
wafferarray = np.ma.masked_where(wafferarray == 0, wafferarray)
np.savetxt("C:/Users/saira/Google Drive/ResearchPapers/Fall2020/Code/pvfigures/waffer/"+parameter+".csv", wafferarray, delimiter=',')
# cmap = matplotlib.cm.YlOrRd
cmap = matplotlib.cm.viridis
cmap.set_bad(color='white')
im = ax.imshow(wafferarray, cmap=cmap, aspect='auto', interpolation='none')
# ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

ax.set_xlabel("Wafer FSR 20nm " + parameter)

fig.colorbar(im, orientation='vertical')
fig.savefig("C:/Users/saira/Google Drive/ResearchPapers/Fall2020/Code/pvfigures/waffer/"+parameter+'.png', dpi=1000)
fig.show()
