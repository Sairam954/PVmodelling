import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib


def waferStyleVisualization(map,no_of_row,no_of_columns,xlabel,ylabel,title):
    fig, ax = plt.subplots()
    map = map.reshape(no_of_row,no_of_columns)
    waffer_map = np.ma.masked_where(map == 0, map)
    cmap = matplotlib.cm.YlOrRd
    cmap.set_bad(color='white')
    im = ax.imshow(waffer_map, cmap=cmap, aspect='auto', interpolation='none')
    # ax.set_xticks(np.arange(0, 240, 24));
    # ax.set_yticks(np.arange(0, 16, 1));
    ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)
    fig.colorbar(im, orientation='vertical')


    ax.set_title(title)
    plt.savefig('results/'+title+'.png', dpi=1000)

    plt.show()

df = pd.read_csv('results/dievariation100by100.csv')
# map = df['max_supported_nlamda'].to_numpy()
# map = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
map = np.array([2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0])
waferStyleVisualization(map,8,10,"Median of Resolution Across Dies",'30mm X 15mm Wafer','PV Resolution each block 100um X 100um')
print(df['resolution'])