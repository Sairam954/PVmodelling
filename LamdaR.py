import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#LamdaR for radial Variation
x = np.linspace(0,300,240)
y = np.linspace(0,300,16)
X, Y = np.meshgrid(x,y)
a3 = 0.0007185921996176913
a4 = 1.7891324575070937
a5 = 1.0047564636509498
f_radial = a3*np.cos(a4*np.sqrt(X**2+Y**2)+a5)
distribution = f_radial.flatten()
distribution = np.interp(distribution,(distribution.min(),distribution.max()),(-0.5,0.5))
waffer_map = f_radial.reshape(16,240)
#
# Q_distribution = np.random.normal(6000,6000*0.0940,3168)
# # Q_distribution = Q_distribution.reshape(66,48)
# waffer_map = np.zeros((16,240))
empty_rows = [0,1,2,3,12,13,14,15]
empty_columns = [x for x in range(240) if x != 50]

i =0
# print([x for x in range(48)])
# print([x for x in range(168,240)])
for row in range(16):
    for column in range(240):
        if (row in empty_rows) and (column in [x for x in range(48)] or column in [x for x in range(168,240)]):
            waffer_map[row][column] = 0
            i+=1
        else:
          waffer_map[row][column] = distribution[i]
          i+=1


# print(waffer_map[0][0])
fig, ax = plt.subplots()
waffer_map = np.ma.masked_where(waffer_map == 0, waffer_map)
cmap = matplotlib.cm.viridis
cmap.set_bad(color='white')
im = ax.imshow(waffer_map,cmap=cmap, aspect='auto', interpolation='none')

# draw gridlines
# ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
ax.set_xticks(np.arange(0, 240, 24));
ax.set_yticks(np.arange(0, 16, 1));
fig.colorbar(im, orientation='vertical')
plt.show()




#  LamdaR for leveling variation

# x = np.linspace(0,300,240)
# y = np.linspace(0,300,16)
# X, Y = np.meshgrid(x,y)
# a1 = -0.012
# a2 = 0.0189
# f_leveling = a1*X + a2*Y
# distribution = f_leveling.flatten()
# distribution = np.interp(distribution,(distribution.min(),distribution.max()),(-0.5,0.5))
# waffer_map = f_leveling.reshape(16,240)
# #
# # Q_distribution = np.random.normal(6000,6000*0.0940,3168)
# # # Q_distribution = Q_distribution.reshape(66,48)
# # waffer_map = np.zeros((16,240))
# empty_rows = [0,1,2,3,12,13,14,15]
# empty_columns = [x for x in range(240) if x != 50]
#
# i =0
# # print([x for x in range(48)])
# # print([x for x in range(168,240)])
# for row in range(16):
#     for column in range(240):
#         if (row in empty_rows) and (column in [x for x in range(48)] or column in [x for x in range(168,240)]):
#             waffer_map[row][column] = 0
#             i+=1
#         else:
#           waffer_map[row][column] = distribution[i]
#           i+=1
#
#
# # print(waffer_map[0][0])
# fig, ax = plt.subplots()
# waffer_map = np.ma.masked_where(waffer_map == 0, waffer_map)
# cmap = matplotlib.cm.viridis
# cmap.set_bad(color='white')
# im = ax.imshow(waffer_map,cmap=cmap, aspect='auto', interpolation='none')
#
# # draw gridlines
# # ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
# ax.set_xticks(np.arange(0, 240, 24));
# ax.set_yticks(np.arange(0, 16, 1));
# fig.colorbar(im, orientation='vertical')
# plt.show()
import pandas as pd

columns = ['x','y','lamda']
# df = pd.DataFrame(columns = columns)
df= pd.read_csv('Radial_datapoints_1.csv',delimiter=',')
# lamdaR = []
# for y in np.arange(0, 37.5,0.25):
#     for x in np.arange(0,300,0.25):
#
#         lamda = -0.12
#         df = df.append({'x':x,'y':y,'lamda':lamda} ,ignore_index= True)
# print("Row 1")
# for y in np.arange(37.5,75,0.25):
#     for x in np.arange(0, 60,0.25):
#         lamda = -0.12
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#     for x in np.arange(60, 240,0.25):
#         lamda = 0.1
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#     for x in np.arange(240, 300,0.25):
#         lamda =  -0.12
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
# print("Row 2")
# for y in np.arange(75,112.5,0.25):
#     for x in np.arange(0, 30,0.25):
#         lamda = -0.12
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#     for x in np.arange(30, 90,0.25):
#         lamda = 0.1
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#     for x in np.arange(90, 210,0.25):
#         lamda =  0.02
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#     for x in np.arange(210, 240,0.25):
#         lamda =  0.1
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#     for x in np.arange(240, 300,0.25):
#         lamda =  -0.12
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
# print("Row 3")
# for y in np.arange(112.5,150,0.25):
#     for x in np.arange(0, 30):
#         lamda = -0.12
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#     for x in np.arange(30, 90,0.25):
#         lamda = 0.1
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#     for x in np.arange(90, 120,0.25):
#         lamda =  0.02
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#     for x in np.arange(120, 180,0.25):
#         lamda = -0.01
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#     for x in np.arange(180, 210,0.25):
#         lamda = 0.02
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#
#     for x in np.arange(210, 240,0.25):
#         lamda =  0.1
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#     for x in np.arange(240, 300,0.25):
#         lamda =  -0.12
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
# print("Row 4")
# for y in np.arange(150,187.5,0.25):
#     for x in np.arange(0, 30,0.25):
#         lamda = -0.12
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#     for x in np.arange(30, 90,0.25):
#         lamda = 0.1
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#     for x in np.arange(90, 120,0.25):
#         lamda =  0.02
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#     for x in np.arange(120, 180,0.25):
#         lamda = -0.01
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#     for x in np.arange(180, 210,0.25):
#         lamda = 0.02
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#
#     for x in np.arange(210, 240,0.25):
#         lamda =  0.1
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#     for x in np.arange(240, 300,0.25):
#         lamda =  -0.12
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
# print("Row 5")
# for y in np.arange(187.5,225,0.25):
#     for x in np.arange(0, 30,0.25):
#         lamda = -0.12
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#     for x in np.arange(30, 90,0.25):
#         lamda = 0.1
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#     for x in np.arange(90, 210,0.25):
#         lamda =  0.02
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#     for x in np.arange(210, 240,0.25):
#         lamda =  0.1
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#     for x in np.arange(240, 300,0.25):
#         lamda =  -0.12
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
# print("Row 6")
# for y in np.arange(225,262.5,0.25):
#     for x in np.arange(0, 60,0.25):
#         lamda = -0.12
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#     for x in np.arange(60, 240,0.25):
#         lamda = 0.1
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
#     for x in np.arange(240, 300,0.25):
#         lamda =  -0.12
#         df = df.append({'x': x, 'y': y, 'lamda': lamda}, ignore_index=True)
# print("Row 7")
# for y in np.arange(262.5, 300,0.25):
#     for x in np.arange(0,300,0.25):
#
#         lamda = -0.12
#         df = df.append({'x':x,'y':y,'lamda':lamda} ,ignore_index= True)
# print("Row 8")
#
# print(df.head())
df.to_csv("Radial_datapoints_1.csv",index = False)
waffer_map = np.zeros((16,240))
empty_rows = [0,1,2,3,12,13,14,15]
empty_columns = [x for x in range(240) if x != 50]

lamda = df[(df['x'] == float(0)) & (df['y'] == float(0))]['lamda']
# print(lamda.head())
# print('Lamda Value',lamda.values[0] )

i =0
# print([x for x in range(48)])
# print([x for x in range(168,240)])
# this loop runs in blocks
block_to_xcordinate = 1.25
block_to_ycordinate = 18.75
for row in range(16):
    for column in np.arange(0,240):
        if (row in empty_rows) and (column in [x for x in range(48)] or column in [x for x in range(168,240)]):
            waffer_map[row][column] = 0
            i+=1
        else:
          lamda =df[(df['y'] == float(row*block_to_ycordinate)) & (df['x'] == float(column*block_to_xcordinate))]['lamda']
          print("row",row)
          print("column", column)

          print("Lamda",lamda.values[0])
          waffer_map[row][column] = lamda.values[0]
          i+=1


print(waffer_map[0][0])

fig, ax = plt.subplots()
waffer_map = np.ma.masked_where(waffer_map == 0, waffer_map)
cmap = matplotlib.cm.viridis
cmap.set_bad(color='white')
im = ax.imshow(waffer_map,cmap=cmap, aspect='auto', interpolation='none', vmin=-0.5, vmax = 0.5)

# draw gridlines
# ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
ax.set_xticks(np.arange(0, 240, 24));
ax.set_yticks(np.arange(0, 16, 1));
fig.colorbar(im, orientation='vertical')
plt.show()
# df.to_csv("Radial_datapoints.csv",index = False)

