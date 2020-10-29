import numpy as np
import matplotlib.pyplot as plt
import matplotlib


parameters = ['FINESSE']
parmeters_mean = {'Q':6000,
                  'ER':10,
                  'LAMDA_R':1550}
parameters_std_intra = {'Q':(0.076144+0.094),#(intra std+ residual std)
                  'ER':(0.070406+0.0891),#(intra std+ residual std)
                  'LAMDA_R':0.17,#(Local variation std)
                        'FINESSE':(0.076144+0.094)}
parameters_std_inter = {'Q':0.06536,#(inter std)
                  'ER':0.06049,#(inter std)
                  'LAMDA_R':(0.33+1.40),#(Global +TXRX)
                        'FINESSE':0.06536}

Q_MEAN = 6500
FSR = 20 *1e-9
LAMDA_R = 1550* 1e-9
parmeters_mean['FINESSE'] = (Q_MEAN*FSR)/LAMDA_R
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

for parameter in parameters:
       print("Parameter",parameter)
       mean = parmeters_mean[parameter]
       std_intra = parameters_std_intra[parameter]
       xpos = [0]
       if parameter=='LAMDA_R':
              distribution = np.random.normal(mean, std_intra, int(no_of_blocks))

       else:
              distribution = np.random.normal(mean, mean*std_intra, int(no_of_blocks))
       waffer_map = distribution.reshape(int(die_xdim / block_xdim), int(die_ydim / block_ydim))
       np.savetxt('intrapvfolder/'+parameter+'/'+folder_name+'IntraPV_'+parameter+'_block'+str(block_xdim)+'_'+str(block_ydim)+'.csv', waffer_map, delimiter=',')
       fig, ax = plt.subplots()
       waffer_map = np.ma.masked_where(waffer_map == 0, waffer_map)
       cmap = matplotlib.cm.YlOrRd
       cmap.set_bad(color='white')
       im = ax.imshow(waffer_map, cmap=cmap, aspect='auto', interpolation='none')
       # ax.set_xticks(np.arange(0, 240, 24));
       # ax.set_yticks(np.arange(0, 16, 1));
       ax.set_xlabel("Intra Die "+parameter)
       fig.colorbar(im, orientation='vertical')

       if(parameter=='FINESSE'):
              title = "Q={q}, FSR={fsr}nm, Finesse={finesse:.2f}, Lamda_R= 1550nm  ".format(
                     q=Q_MEAN, fsr=20,
                     finesse=parmeters_mean['FINESSE'])
              ax.set_title(title)
       else:
              ax.set_title('Block_Size ' + str(block_xdim) + 'mm' + ' X ' + str(block_ydim) + 'mm')
       plt.savefig('intrapvfolder/'+parameter+'/'+folder_name+'/IntraPV_'+parameter+'_block'+str(block_xdim)+'_'+str(block_ydim)+'.png',dpi=1000)

       plt.show()

       no_of_dies = 80
       std_inter_die = parameters_std_inter[parameter]
       inter_die_map = []
       for die in range(no_of_dies):
              waffer_map_inter_die = []
              for block_value in range(no_of_blocks):
                     if parameter == 'LAMDA_R':
                            waffer_map_inter_die.append(
                                   np.random.normal(distribution[block_value],
                                                    std_inter_die, 1)[0])
                     else:
                            waffer_map_inter_die.append(
                                   np.random.normal(distribution[block_value], distribution[block_value] * std_inter_die, 1)[0])
              inter_die_map.append(waffer_map_inter_die)

       # creating csv's
       for die in range(no_of_dies):
              print("Die",die)
              map = np.array(inter_die_map[die]).reshape(int(die_xdim / block_xdim), int(die_ydim / block_ydim))
              np.savetxt('interpvfolder/'+parameter+'/'+folder_name+'/InterDie_' +parameter+'_'+ str(die) +'_block'+str(block_xdim)+'_'+str(block_ydim)+".csv", map, delimiter=',')
              fig, ax = plt.subplots()
              waffer_map = np.ma.masked_where(waffer_map == 0, waffer_map)
              cmap = matplotlib.cm.YlOrRd
              cmap.set_bad(color='white')
              im = ax.imshow(map, cmap=cmap, aspect='auto', interpolation='none')
              # ax.set_xticks(np.arange(0, 240, 24));
              # # ax.set_yticks(np.arange(0, 16, 1));
              ax.set_xlabel("Inter Die " + parameter)
              if (parameter == 'FINESSE'):
                     title = "Q={q}, FSR={fsr}nm, Finesse={finesse:.2f}, Lamda_R= 1550nm  ".format(
                            q=Q_MEAN, fsr=20,
                            finesse=parmeters_mean['FINESSE'])
                     ax.set_title(title)
              else:
                     ax.set_title('Block_Size ' + str(block_xdim) + 'mm' + ' X ' + str(block_ydim) + 'mm')
              # ax.set_title('Block_Size ' + str(block_xdim) + 'mm' + ' X ' + str(block_ydim) + 'mm')
              fig.colorbar(im, orientation='vertical')
              plt.savefig('pvfigures/'+parameter+'/'+folder_name+'/InterDie_'+parameter+ str(die)+'_'+'_block'+str(block_xdim)+'_'+str(block_ydim)+'.png',dpi=1000)
              plt.show()
              # print("Plot Inter")
              # print("Parameter",parameter)

       print("Finished Parameter", parameter)


# ax.bar(xpos, mean,
#        yerr=std,
#        align='center',
#        alpha=0.5,
#        ecolor='black',
#        capsize=10)
# ax.set_ylabel(' Q value')
# ax.set_xticks(xpos)
# # ax.set_xticklabels(labels)
# ax.set_title('Process Variation')
# plt.show()
#


# #ploting all the inter die variation
# for die in no_of_dies:
#        map = inter_die_map.reshape(int(die_xdim/block_xdim),int(die_ydim/block_ydim))




