## Libraries
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

## Open file
Data = pd.ExcelFile('N_E_Data.xls')
Node_Data = pd.read_excel(Data, 'Nodes')
Element_Data = pd.read_excel(Data, 'Elements')


## Functions
# Node Function
def Nodefun(len_num):
        
    Dict = {
            "ID" :     int(Node_Data['Node'][len_num]),
            'X' :        [(Node_Data['X'][len_num]),
                          (Node_Data['Y'][len_num]),
                          (Node_Data['Z'][len_num])],
            "Load" :     [(Node_Data['FX'][len_num]),
                          (Node_Data['FY'][len_num]),
                          (Node_Data['FZ'][len_num])],
            "Moment" :   [(Node_Data['MX'][len_num]),
                          (Node_Data['MY'][len_num]),
                          (Node_Data['MZ'][len_num])],
            "Disp" :     [(Node_Data['dx'][len_num]),
                          (Node_Data['dy'][len_num]),
                          (Node_Data['dz'][len_num])],
            "Rotation" : [(Node_Data['rx'][len_num]),
                          (Node_Data['ry'][len_num]),
                          (Node_Data['rz'][len_num])]
    }
    return Dict

# Element Function
def Elementfun(len_num):
    Dict = {
            "ID" :          int(Element_Data['Element'][len_num]),
            "nodes" :      [int(Element_Data['Node 1'][len_num]),
                            int(Element_Data['Node 2'][len_num])],
            "YoungsMod" : float(Element_Data['E'][len_num]),
            "I" :          [float(Element_Data['Iy'][len_num]),
                            float(Element_Data['Iz'][len_num])],
            "G" :         float(Element_Data['G'][len_num]),
            "A" :         float(Element_Data['A'][len_num]),
            "J" :         float(Element_Data['J'][len_num]),
            "Rho" :       float(Element_Data['Density'][len_num])
    }
    return Dict

## Getting Data in Json Format
# Node Data
nnum = len(Node_Data['Node'])
node = []
for i in range(0,nnum):
    node.append(Nodefun(i))

# Element Data    
enum = len(Element_Data['Element'])
bar = []
for i in range(0,enum):
    bar.append(Elementfun(i))
    
PythontoJson = {
        "node" : list(node),
        "bar" : list(bar)
        }    

##  Output File
# Converting numpy int/float into python int/float
def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

# Outputing Data into Json Format    
with open('result.json','w') as fp:
    json.dump(PythontoJson, fp, indent = 4, default=np_encoder)


## Displays Structure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ScaleLoads=0.00002
Node = PythontoJson['node']

for i in PythontoJson['bar']:

#Plot Original truss	
	ax.plot( \
	[Node[i['nodes'][0]]['X'][0], Node[i['nodes'][1]]['X'][0]], \
	[Node[i['nodes'][0]]['X'][1], Node[i['nodes'][1]]['X'][1]], \
	[Node[i['nodes'][0]]['X'][2], Node[i['nodes'][1]]['X'][2]], \
	color='k',linewidth=3, marker='o',markerfacecolor='red',
    markersize=8)

#plot Loads
for i in PythontoJson['node']:
    try :
        ax.quiver( i['X'][0],i['X'][1],i['X'][2],\
	    ScaleLoads*i['Load'][0],ScaleLoads*i['Load'][1],ScaleLoads*i['Load'][2],\
	    color='g',arrow_length_ratio=1,linewidth=3)
    except : 
        ax.quiver( i['X'][0],i['X'][1],i['X'][2],\
	    0,0,0,\
	    color='g',arrow_length_ratio=0.01,linewidth=3)

ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$Z$')
ax.legend()
plt.show()