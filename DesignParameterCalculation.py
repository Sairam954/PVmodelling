import numpy as np

def getFSR(lamda_r,ring_radius,ng):
    return (np.square(lamda_r)/(2*np.pi*ring_radius*ng))


# find the design parameter values from the paper 2017 Date Energy-Performance Optimized Design of SiliconPhotonic Interconnection Networks for High-Performance Computing
lamda_r = 1550 #nm
ring_radius = [5000, 10000, 20000] #nm
L = []
T = [0.93, 0.87, 0.77]
ng = 4.2
FSR = []
for r in ring_radius:
    FSR.append(getFSR(lamda_r,r,ng))
    L.append(10**(2*np.pi*rx))
    # print(getFSR(lamda_r, r, ng))

delta_lamda_3db = []
print(L)

for l, t, fsr in zip(L, T, FSR):
    delta_lamda_3db.append((fsr/np.pi)*np.arccos((1-np.square(1-t*np.sqrt(l*1e-9)))/(2*t*np.sqrt(l*1e-9))))
    # print((fsr/np.pi)*np.arccos((1-np.square(1-t*np.sqrt(l*1e-9)))/(2*t*np.sqrt(l*1e-9))))
Q = []
for delta in delta_lamda_3db:
      Q.append(lamda_r/delta)
      print(lamda_r/delta)




for t,l in zip(T, L):
    # alpha = np.sqrt(np.exp(1*1e-2*l*1e-9))
    # alpha = t
    # A = 1 - np.square((t-np.exp(-alpha*l*1e-9))/(1-t*np.exp(-alpha*l*1e-9)))
    # print("A ",A)
    # ER = (1/(1-A))
    # print("ER",ER)
    # print("ER dB",10*np.log10(ER))
    trmin = np.square((t - np.sqrt(l)) / (1 - t * np.sqrt(l)))
    trmax = 1
    ER = 1/trmin
    print(ER)
