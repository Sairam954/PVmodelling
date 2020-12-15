
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
class Weight(object):
    def __init__(self):
        T0 = None
        t1,t2 = 0,0
        L = None
        FSR = None
        FWHM = None


    def getWeightMR(self,phase,finesse):
        critical_coupling = True

        if(critical_coupling):
            T0 =0
        a = (2 * finesse) / np.pi
        t_of_phase = (np.square(a * np.sin(phase / 2))) / (1 + np.square(a * np.sin(phase / 2)))
        t_of_pi = (np.square(a)) / (1 + np.square(a))
        weight = t_of_phase/t_of_pi
        return weight
    def getPhaseShift(self,weight,finesse):
        a = (2*finesse)/np.pi
        phase = 2*np.arcsin(np.sqrt(weight/(1+pow(a,2)-weight*pow(a,2))))
        return phase

    def getSensitivity(self,phase,finesse):
        a = (2 * finesse) / np.pi
        sensitivity = ((1+pow(a,2))*(0.5*pow(a,2)*np.sin(phase)))/(pow(a,2)*pow((1+pow(a*np.sin(phase/2),2)),2))
        return sensitivity
    def getMaxPhase(self,finesse):
        a = (2 * finesse) / np.pi
        numerator = 3*pow(a,2)+2-np.sqrt(9*pow(a,4)+4*pow(a,2)+4)
        denom = pow(a,2)- 2+ np.sqrt(9*pow(a,4)+4*pow(a,2)+4)
        max_phase = 2*np.arctan(np.sqrt(numerator/denom))
        return max_phase
    def getMaxSensitivity(self,finesse):
        return 0.2067*finesse
    def getHeatDiffusion(self,ring_radius,r_chip_boundary,r_source):
        heat_diffusion = (np.log(r_chip_boundary / r_source) / np.log(r_chip_boundary / ring_radius))
        return heat_diffusion
    def getMaxThermalCrossTalkError(self,heater_thickness,ring_radius,r_chip_boundary,r_source,finesse):

        # for source in r_source:
        #     heat_diffusion = heat_diffusion+(np.log(r_chip_boundary/source)/np.log(r_chip_boundary/ring_radius))

        heat_diffusion = (np.log(r_chip_boundary / r_source) / np.log(r_chip_boundary / ring_radius))
        # we consider in an array if MR adjacent two MRs incur thermal losses so summation of the sources
        # considetring both the rings are at a similar or equal distance
        # heat_diffusion = 2*heat_diffusion
        max_error = 0.65*finesse*(heater_thickness/(2*ring_radius))*heat_diffusion
        return max_error
    def getThroughPowerRatioOrEta(self,finesse,nlamda,t0=0):
        b = (4*np.square(finesse))/(np.square(np.pi))
        a = np.divide((1-np.cos(2*np.pi/nlamda)),(2/b))

        eta_numerator = t0*(1+b)+a*(1+b)
        eta_denominator = (t0+b)*(a+1)
        eta = eta_numerator/eta_denominator
        return eta

weightObj = Weight()
finesse = [10,20,50,100]
fig , axs = plt.subplots(3,2)

for f in finesse:
    weightfactor = []
    weight = []
    phases = np.linspace(0,0.5,500)
    for phase in phases:
       # print(phase*0.1, weightObj.getWeightMR(phase*0.1*np.pi,f))
       weightfactor.append(phase)
       weight.append(weightObj.getWeightMR(phase*np.pi,f))
    axs[0,0].plot(weightfactor, weight, '-')


axs[0,0].set_xlabel("Weight Factor")
axs[0,0].set_ylabel("Weight")



weights = np.linspace(0,1,1000)
for f in finesse:
    sensitivity = []
    for weight in weights:
        sensitivity.append(weightObj.getSensitivity(weightObj.getPhaseShift(weight,f),f))
    axs[0,1].plot(weights,sensitivity,'-')

axs[0,1].set_xlabel("Weight")
axs[0,1].set_ylabel("Sensitivity")

maxweightfactor = []
finesses = np.linspace(0,100,100)
for f in finesses:
    maxweightfactor.append(weightObj.getMaxPhase(f))

axs[1,0].plot(finesses, np.array(maxweightfactor)/np.pi, '-')
axs[1,0].set_xlabel("Finesse")
axs[1,0].set_ylabel("Max Weight Factor")

maxsensitivity = []
for f in finesses:
    maxsensitivity.append(weightObj.getMaxSensitivity(f))
axs[1, 1].plot(finesses, maxsensitivity, '-')
axs[1,1].set_ylabel("Max Sensitivity")
axs[1,1].set_xlabel("Finesse")

crosstalk_source_distance = np.linspace(10,1000,100)
heatdiffusion = []
for source_distance in crosstalk_source_distance:
    heatdiffusion.append(weightObj.getHeatDiffusion(10*1e-6,1*1e-3,source_distance*1e-6))

axs[2,0].plot(crosstalk_source_distance, heatdiffusion, '-')
axs[2,0].set_ylabel("Heat Diffusion")
axs[2,0].set_xlabel("Cross Talk Source ")


mrr_pitch = np.linspace(100, 990, 300)
resolution = []
heater_thickness = 100*1e-9
ring_radius = 10*1e-6
chip_boundary_radius = 1*1e-3
finesses = np.linspace(20, 100, 300)
res = []
for finesse in finesses:
    for pitch in mrr_pitch:

        dict = {}
        thermal_crosstalk_error = weightObj.getMaxThermalCrossTalkError(heater_thickness, ring_radius, chip_boundary_radius, r_source=(pitch*1e-6), finesse=finesse)
        # print("Thermal_Crosstalk_error", thermal_crosstalk_error,"Pitch", pitch,"Finesse",finesse,"Resolution",1/thermal_crosstalk_error)

        dict["Pitch"] = pitch
        dict["Finesse"] = finesse
        dict["Thermal CrossTalk"] = thermal_crosstalk_error
        dict["Resolution"] = int(1/(thermal_crosstalk_error)).bit_length()
        res.append(int(1/(thermal_crosstalk_error)).bit_length())

        resolution.append(dict)

pitch_range = np.linspace(100, 990, 300)
finesee_range = np.linspace(20, 100, 300)
X, Y = np.meshgrid(pitch_range, finesee_range)
resolution_range = np.asarray(res).reshape(300,300)
# max_thermal_crosstalk_range = weightObj.getMaxThermalCrossTalkError(heater_thickness, ring_radius, chip_boundary_radius, X*1e-6, Y)
# resolution_range = 1/max_thermal_crosstalk_range
# resolution_range = resolution_range.astype(int)
# resolution_range = np.ceil(np.log2(resolution_range)).astype(int)
# print(resolution_range.shape)
# print(resolution_range)
contour = axs[2, 1].contour(X, Y, resolution_range)
axs[2, 1].clabel(contour,fmt = '%2d', inline=True, fontsize=8)
plt.show()
df = pd.DataFrame(resolution)
df.to_csv("thermalcrosstalk.csv",index=False)
plt.show()
finesse_x = np.linspace(50,200,1000)
nlamda_y = np.linspace(50,200,1000)

X,Y = np.meshgrid(finesse_x,nlamda_y)
# print(X.shape)
# print(Y.shape)
Z = weightObj.getThroughPowerRatioOrEta(X.ravel(), Y.ravel())
Z = Z.reshape(X.shape)
# print(Z.shape)

# print(weightObj.getThroughPowerRatioOrEta(50,200))

fig2 = plt.figure()
ax = plt.axes(projection = '3d')
ax.contour3D(X,Y,Z,100, cmap ='viridis')
ax.set_xlabel('Finesse')
ax.set_ylabel('Nlamda')
ax.set_zlabel('Through Power Ratio Eta')
ax.view_init(25,-250)
fig2
plt.show()