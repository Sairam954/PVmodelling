import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

die_xdim = 30 #mm
die_ydim = 37.5 #mm

ring_radius = 5*1e-3 #mm
pitch = 5*1e-3 #mm

# block_xdim = 2*(ring_radius)+2*(pitch)
# block_ydim = 2*(ring_radius)+2*(pitch)
block_xdim = 0.2 #mm
block_ydim = 0.2 #mm
folder_name = 'Block2by2'

no_of_blocks = int(die_xdim/block_xdim)*int(die_ydim/block_ydim)

Q = [2500, 6000, 10000]
FSR = [10, 20, 50, 80] #nm
LAMDA_R = 1550 #nm

Q_STD = 0.14026
FINESSE = []
configuration_list = []
parameter = 'FINESSE'
for q in Q:
    for f in FSR:
        configuration = {}
        finesse = (q*f)/LAMDA_R
        FINESSE.append(finesse)
        configuration['Q'] = q
        configuration['FSR'] = f
        configuration['FINESSE'] = finesse
        configuration_list.append(configuration)

finesse_distribution = []

for confid in range(len(configuration_list)):
    finesse_distribution.append(np.random.normal(configuration_list[confid]['FINESSE'],configuration_list[confid]['FINESSE']*Q_STD, int(no_of_blocks)))

    xmin = finesse_distribution[confid].min()
    xmax = finesse_distribution[confid].max()
    mean = finesse_distribution[confid].mean()
    std = Q_STD
    plt.hist(finesse_distribution[confid], bins=25, density=True, alpha=0.6, color='g')
    x = np.linspace(xmin,xmax,100)
    p = norm.pdf(x, mean, configuration_list[confid]['FINESSE']*std)
    title = "Q={q}, FSR={fsr}nm, Finesse={finesse:.2f}, Lamda_R= 1550nm  ".format(q =configuration_list[confid]['Q'] , fsr = configuration_list[confid]['FSR'], finesse= configuration_list[confid]['FINESSE'])
    plt.title(title)
    plt.plot(x, p, 'k', linewidth=2)
    # plt.hist(finesse_distribution[0])
    plt.savefig('pvfigures/' + parameter + '/' + folder_name + '/PV_' + parameter+'Q_'+str(configuration_list[confid]['Q'])+'_FSR_'+str(configuration_list[confid]['FSR'])+'.png', dpi=1000)

    plt.show()

