
import matplotlib.pyplot as plt
import json
from mpl_toolkits.mplot3d import Axes3D

inputFile='result.json.out'
with open(inputFile) as data_file:
    jsonToPython = json.load(data_file)
    
x = 0
y = 1
z = 2

# ============================ Data input complete ============================

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ScaleDisps=10
ScaleLoads=0.00000005
Node=jsonToPython['node']
ax.set_axis_off()

for i in jsonToPython['bar']:

#Plot Original truss	
	ax.plot( \
	[Node[i['nodes'][0]]['X'][x], Node[i['nodes'][1]]['X'][x]], \
	[Node[i['nodes'][0]]['X'][y], Node[i['nodes'][1]]['X'][y]], \
	[Node[i['nodes'][0]]['X'][z], Node[i['nodes'][1]]['X'][z]], \
	color='b',linewidth=3,linestyle='dashed',marker='o',markerfacecolor='red',
    markersize=8)

#Plot Deformed truss		
	ax.plot( \
	[Node[i['nodes'][0]]['X'][x]+ScaleDisps*Node[i['nodes'][0]]['Disp'][x], \
	Node[i['nodes'][1]]['X'][x]+ScaleDisps*Node[i['nodes'][1]]['Disp'][x]], \
	[Node[i['nodes'][0]]['X'][y]+ScaleDisps*Node[i['nodes'][0]]['Disp'][y], \
	Node[i['nodes'][1]]['X'][y]+ScaleDisps*Node[i['nodes'][1]]['Disp'][y]], \
	[Node[i['nodes'][0]]['X'][z]+ScaleDisps*Node[i['nodes'][0]]['Disp'][z], \
	Node[i['nodes'][1]]['X'][z]+ScaleDisps*Node[i['nodes'][1]]['Disp'][z]], \
	color='k',linewidth=3,marker='o',markerfacecolor='red',markersize=8)
print('black:\tundeformed truss')
print('blue:\tdeformed truss')

for i in jsonToPython['node']:
#plot displacements
	ax.quiver( i['X'][x],i['X'][y],i['X'][z],\
	ScaleDisps*i['Disp'][0],ScaleDisps*i['Disp'][1],ScaleDisps*i['Disp'][2],\
	color='g',linestyle='dashed',arrow_length_ratio=0.0001)
	
#plot Loads
	ax.quiver( i['X'][x],i['X'][y],i['X'][z],\
	ScaleLoads*i['Load'][0],ScaleLoads*i['Load'][1],ScaleLoads*i['Load'][2],\
	color='r',arrow_length_ratio=1,linewidth=3)
print('green:\tdisplacements')
print('red:\tloads')

print('displacement scale factor:\t',ScaleDisps)
print('load scale factor:\t',ScaleLoads)

# Label can be turned on if Line 21 is turned off
#ax.set_xlabel('$X$')
#ax.set_ylabel('$Y$')
#ax.set_zlabel('$Z$')
#ax.set_zlim(-1,1)

ax.legend()
plt.show()


