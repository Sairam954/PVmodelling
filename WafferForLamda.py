import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import os
from os import listdir
from os.path import isfile, join

die_data_path = "C://Users//saira//Google Drive//ResearchPapers//Fall2020//Code//interpvfolder//LAMDA_R//Block2by2//"

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
leveling_std = 1.46 #nm

empty_columns_for_row = {0: [0, 1, 7, 8, 9],
                         1: [0, 9],
                         6: [0, 9],
                         7: [0, 1, 7, 8, 9]}
parameter = 'LAMDA_R'
row_std_coeff = 3
row_std_dict = {}
i = no_of_row - 1
# j = no_of_columns-1
while i >= 0:
    # for i in range(0, no_of_row):
    blockrowstartidx = i * die_x_dim
    stdlist = []
    if (i != 0 and i % 2 == 0):
        row_std_coeff = row_std_coeff - 0.75
    column_std = row_std_coeff

    print("blockrowstartidx  :", blockrowstartidx)

    # while j>=0:
    for j in range(0, no_of_columns):
        print("Die :", die_number)
        columnstartidx = j * die_y_dim
        std = column_std * leveling_std
        if (i in empty_rows) and (j in empty_columns_for_row[i]):
            # print("Row Range= " + str(blockrowstartidx) + ' ' + str(blockrowstartidx + die_x_dim))
            # print("Column Range " + str(columnstartidx) + ' ' + str(columnstartidx + die_y_dim))
            wafferarray[blockrowstartidx:(blockrowstartidx + die_x_dim),columnstartidx:(columnstartidx + die_y_dim)] = 0
            die_number += 1
        else:

            # print("Row Range= " + str(blockrowstartidx) + ' ' + str(blockrowstartidx + die_x_dim))
            # print("Column Range " + str(columnstartidx) + ' ' + str(columnstartidx + die_y_dim))
            # print(wafferarray[blockrowstartidx:(blockrowstartidx+die_x_dim),columnstartidx:(columnstartidx+die_y_dim)].shape)
            distribution = pd.read_csv(files[die_number]).to_numpy()
            distribution_mean = distribution.mean()
            # print("Mean" + str(distribution_mean))
            deviation = std
            # if (i >= j):

            # distribution = np.random.normal(distribution_mean, 3*leveling_std,die_x_dim*die_y_dim )
            # distribution = distribution.reshape(die_x_dim,die_y_dim)
            distribution = distribution + deviation
            wafferarray[blockrowstartidx:(blockrowstartidx + die_x_dim),columnstartidx:(columnstartidx + die_y_dim)] = distribution
            # else:
            #     distribution = distribution-deviation
            # distribution = np.random.normal(distribution_mean, -3 * leveling_std, die_x_dim*die_y_dim)
            # distribution = distribution.reshape(die_x_dim, die_y_dim)
            die_number += 1
        if (i % 2 == 0):  # even row change coeffiecient at even rows
            if (j != 0 and j % 2 == 0):
                column_std = column_std - 0.75
        else:
            if (j != 0 and j % 2 == 1):
                column_std = column_std - 0.75
        stdlist.append(column_std)
        # j-=1
    row_std_dict[i] = stdlist
    i -= 1
print(row_std_dict)
fig, ax = plt.subplots()
wafferarray = np.ma.masked_where(wafferarray == 0, wafferarray)
np.savetxt("C:/Users/saira/Google Drive/ResearchPapers/Fall2020/Code/pvfigures/waffer/" + parameter + ".csv",
           wafferarray, delimiter=',')
cmap = matplotlib.cm.viridis
cmap.set_bad(color='white')
im = ax.imshow(wafferarray, cmap=cmap, aspect='auto', interpolation='none')
# ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

ax.set_xlabel("Wafer Leveling" + parameter)

fig.colorbar(im, orientation='vertical')
fig.tight_layout()
fig.savefig("C:/Users/saira/Google Drive/ResearchPapers/Fall2020/Code/pvfigures/waffer/" + parameter+'Leveling'+ '.png', dpi=1000)
fig.show()

###############Radial Variations

radial_variation_dict = {7: [-1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1, -1],
                         6: [-1,    -1, -0.75, -0.75, -0.75, -0.75, -0.75, -0.75,    -1, -1],
                         5: [-1, -0.75, -0.75,  -0.25,  -0.25,  -0.25,  -0.25, -0.75, -0.75, -1],
                         4: [-1, -0.75, -0.75,  -0.25, 0.1, 0.1,  -0.25, -0.75, -0.75, -1],
                         3: [-1, -0.75, -0.75,  -0.25, 0.1, 0.1,  -0.25, -0.75, -0.75, -1],
                         2: [-1, -0.75, -0.75,  -0.25,  -0.25,  -0.25,  -0.25, -0.75, -0.75, -1],
                         1: [-1,    -1, -0.75, -0.75, -0.75, -0.75, -0.75, -0.75,    -1, -1],
                         0: [-1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1, -1]}
radial_std = 0.9284 #nm
die_number = 0
i = no_of_row - 1
while i >= 0:
    # for i in range(0, no_of_row):
    blockrowstartidx = i * die_x_dim

    for j in range(0, no_of_columns):
        print("Die :", die_number)
        columnstartidx = j * die_y_dim
        print("Radial Variation ",radial_variation_dict[i][j])
        std = radial_variation_dict[i][j] * leveling_std
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
            distribution = wafferarray[blockrowstartidx:(blockrowstartidx + die_x_dim),columnstartidx:(columnstartidx + die_y_dim)]
            distribution_mean = distribution.mean()
            print("Mean" + str(distribution_mean))
            deviation = std
            print("Deviation",deviation)
            # if (i >= j):

            # distribution = np.random.normal(distribution_mean, 3*leveling_std,die_x_dim*die_y_dim )
            # distribution = distribution.reshape(die_x_dim,die_y_dim)
            distribution = distribution + deviation
            wafferarray[blockrowstartidx:(blockrowstartidx + die_x_dim),
            columnstartidx:(columnstartidx + die_y_dim)] = distribution
            die_number+=1
            # else:
            #     distribution = distribution-deviation
            # distribution = np.random.normal(distribution_mean, -3 * leveling_std, die_x_dim*die_y_dim)
            # distribution = distribution.reshape(die_x_dim, die_y_dim)


    i -= 1
fig, ax = plt.subplots()
wafferarray = np.ma.masked_where(wafferarray == 0, wafferarray)
np.savetxt("C:/Users/saira/Google Drive/ResearchPapers/Fall2020/Code/pvfigures/waffer/" + parameter + ".csv",
           wafferarray, delimiter=',')
cmap = matplotlib.cm.viridis
cmap.set_bad(color='white')
im = ax.imshow(wafferarray, cmap=cmap, aspect='auto', interpolation='none')
# ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

ax.set_xlabel("Wafer Radial " + parameter)

fig.colorbar(im, orientation='vertical')
fig.tight_layout()
fig.savefig("C:/Users/saira/Google Drive/ResearchPapers/Fall2020/Code/pvfigures/waffer/" + parameter +'Radial'+'.png', dpi=1000)
fig.show()