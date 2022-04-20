import numpy as np #numpy (np for short) is a numerical operation library for Python
import json        #json is a normalised file format used for the truss data file 
## ========================== Functions =======================================
# Python Matrix function (Minus 1 in dictionary)
def Matrix(DICT):
    for i in range(0,len(DICT)):
        DICT[i] = DICT[i]-1
    return DICT 


# Adds smaller matrices to big local one (Local ke)
def Stiffness(IN,OUT,IND,LOCAL):
    for i in range(0,len(OUT)):
        for j in range(0,len(OUT)):
            LOCAL[OUT[i],OUT[j]]  += IND[IN[i],IN[j]]
    return LOCAL


# Gather Matrix function
def Gather(IN1,IN2,OUT1,OUT2,IND,LOCAL):
    for i in range(0,len(IN1)):
        for j in range(0,len(IN1)):
            LOCAL[OUT1[i],OUT1[j]]  += IND[IN1[i],IN1[j]]
            LOCAL[OUT1[i],OUT2[j]]  += IND[IN1[i],IN2[j]]
            LOCAL[OUT2[i],OUT1[j]]  += IND[IN2[i],IN1[j]]
            LOCAL[OUT2[i],OUT2[j]]  += IND[IN2[i],IN2[j]]
    return LOCAL


#read input file 
inputFile='beam.json'
with open(inputFile) as data_file:
    jsonToPython = json.load(data_file)

nnodes=len(jsonToPython['node'])
nbars=len(jsonToPython['bar'])
Range=len(jsonToPython['node'][0]['X'])
ndof=Range*2

#Initialisation of matrices, vectors and arrays
nodes=np.zeros((nnodes,Range))
Loads=[['Free' for x in range(Range)] for y in range(nnodes)]
Disps=[['Free' for x in range(Range)] for y in range(nnodes)]
Rotations=[['Free' for x in range(Range)] for y in range(nnodes)]
Moments=[['Free' for x in range(Range)] for y in range(nnodes)]
bars=[[0 for x in range(2)] for y in range(nbars)]
A=np.zeros(nbars)
E=np.zeros(nbars)
J=np.zeros(nbars)
G=np.zeros(nbars)
I=np.zeros((nbars,2))
Rho = np.zeros((nbars))


#Parse nodal properties
j=0
for i in jsonToPython['node']:
    nodes[j]=(i['X'])
    Loads[j]=(i['Load'])
    Disps[j]=(i['Disp'])
    Rotations[j]=(i['Rotation'])
    Moments[j]=(i['Moment'])
    j+=1
	
#Parse bar properties 
j=0
for i in jsonToPython['bar']:
    bars[j]=(i['nodes'])
    A[j]=(i['A'])
    E[j]=(i['YoungsMod'])
    J[j]=(i['J'])
    G[j]=(i['G'])
    I[j]=(i['I'])
    Rho[j]=(i['Rho'])
    j+=1
    
    
## ============================ Data input complete ===========================
	
Dx=np.zeros(ndof)
K = np.zeros((nnodes*ndof,nnodes*ndof))
mass = 0 

for barID in range(0,nbars):
	
    ## Calculate Length of elements ===========================================
    node1ID= int(bars[barID][0])
    node2ID= int(bars[barID][1])
    N2 = nodes[node2ID, :]
    N1 = nodes[node1ID, :] 
    Dx = N2 - N1
    Len = np.sqrt(np.dot(np.transpose(Dx),Dx))
    a = Len/2
    
    ## Calculate ke_t, ke_z, ke_y, TK =========================================
    # ke_t
    ke_t = ((E[barID]*A[barID])/(2*a))*np.array(([1,-1],
                                                 [-1,1]))
        # Truss Stiffness matrix locations
    ke_tl = Matrix({0:1,1:2})
    ke_tb = Matrix({0:1,1:7})            
        
    # Beam Stiffness (ke_z, ke_y)
    ke_y = np.array(([3,3*a,-3,3*a],
                     [3*a,4*pow(a,2),-3*a,2*pow(a,2)],
                     [-3,-3*a,3,-3*a],
                     [3*a,2*pow(a,2),-3*a,4*pow(a,2)]))
    ke_z = np.array(([3,-3*a,-3,-3*a],
                     [-3*a,4*pow(a,2),3*a,2*pow(a,2)],
                     [-3,3*a,3,3*a],
                     [-3*a,2*pow(a,2),3*a,4*pow(a,2)]))
    
    # Beam Stiffness matrix locations in the y directions    
    ke_by = ((E[barID]*I[barID,1])/(2*pow(a,3)))*ke_y
    ke_byl = Matrix({0:1,1:2,2:3,3:4})
    ke_byb = Matrix({0:2,1:6,2:8,3:12})  
    
    # Beam Stiffness matrix locations in the z directions   
    ke_bz = ((E[barID]*I[barID,0])/(2*pow(a,3)))*ke_z
    ke_bzl = Matrix({0:1,1:2,2:3,3:4})
    ke_bzb = Matrix({0:3,1:5,2:9,3:11})   

    # Torsion Stiffness
    Tk = ((G[barID]*J[barID])/(2*a))*np.array(([1,-1],[-1,1]))
        # Torsion stiffness matrix locations
    ke_TKl = Matrix({0:1,1:2})
    ke_TKb = Matrix({0:4,1:10})
    
    ## Calculate Local ke =====================================================
    ke = np.zeros((ndof*2,ndof*2))
    ze = ke
    ke = Stiffness(ke_tl,ke_tb,ke_t,ke)
    ke = Stiffness(ke_bzl,ke_bzb,ke_bz,ke)
    ke = Stiffness(ke_byl,ke_byb,ke_by,ke)
    ke = Stiffness(ke_TKl,ke_TKb,Tk,ke)
    
    ## Transforamtion Matrix ==================================================
    cosx = np.zeros(Range) 
    N21 = Dx 
    
    if N2[0]==N1[0] and N2[1]==N1[1]:
        if N2[2] > N1[2]:
            T3 = np.array(([0,0,1],
                           [0,1,0],
                           [-1,0,0]))
        else:
            T3  = np.array(([0,0,-1],
                            [0,1,0],
                            [1,0,0]))
    else:
        
    ## CosX(x,y,z)
        for i in range(0,Range):
            cosx[i] = N21[i]/(2*a)
        lx = cosx[0]
        mx = cosx[1]
        nx = cosx[2]
  
    ## CosY(x,y,z)
        D = np.sqrt(pow(lx,2)+pow(mx,2))
        ly = -mx/D
        my = lx/D
        ny = 0

    ## CosZ(x,y,z)
        lz = (-lx*ly)/D
        mz = (-mx*my)/D
        nz = D
    ## T3 ===================================================================== 
        T3 = np.array(([lx,mx,nx],
                       [ly,my,ny],
                       [lz,mz,nz]))
    T30 = np.zeros((3,3))
    
    ## T Matrix ===============================================================
    T = np.concatenate(((np.concatenate((T3,T30,T30,T30),axis=1)),
                        (np.concatenate((T30,T3,T30,T30),axis=1)),
                        (np.concatenate((T30,T30,T3,T30),axis=1)),
                        (np.concatenate((T30,T30,T30,T3),axis=1))),axis=0)
    ##
    Ke = np.dot(np.dot(np.transpose(T),ke),T)
    #Gather Matrix
    IN1 = Matrix({0:1,1:2,2:3,3:4,4:5,5:6})
    IN2 = Matrix({0:7,1:8,2:9,3:10,4:11,5:12}) 
    OUT1 = {0:1,1:2,2:3,3:4,4:5,5:6}
    OUT2 = {0:7,1:8,2:9,3:10,4:11,5:12}
    
    for i in range(0,len(OUT1)):
        OUT1[i] = node1ID*ndof+i
        OUT2[i] = node2ID*ndof+i      
    K = Gather(IN1,IN2,OUT1,OUT2,Ke,K)
    
    ## Calculate mass
    rho = 2700
    mass += A[barID]*Len*Rho[barID]
    
## Build the matrices used to solve the truss problem =========================
A=K
u_0 = np.zeros(ndof*nnodes)        
u = np.zeros(ndof*nnodes)
P = np.zeros(ndof*nnodes)
Locations_disp = np.zeros((nnodes,Range))
Locations_rot = np.zeros((nnodes,Range))
nonZeroD=0
nonZeroR=0
for i in range(0,nnodes):
    for j in range(0,Range):
        try:
            u_0[i*Range*2+j] = Disps[i][j]
            u[i*Range*2+j] = Disps[i][j]            
        except:
            u_0[i*Range*2+j] = 0
            u[i*Range*2+j] = 0
            nonZeroD += 1
            Locations_disp[i][j] = 1
        try: 
            u_0[Range*(i*2+1)+j] = Rotations[i][j]
            u[Range*(i*2+1)+j] = Rotations[i][j] 
        except:
            u_0[Range*(i*2+1)+j] = 0
            u[i*Range*2+j] = 0
            nonZeroR += 1
            Locations_rot[i][j] = 1
        try:
            P[i*Range*2+j] = Loads[i][j]
            P[Range*(i*2+1)+j] = Moments[i][j]
        except:
            P[i*Range*2+j] = 0
            P[Range*(i*2+1)+j] = 0    
    
    
 # Enforce essential/Dirichlet/kinetic/displacement boundary conditions by Gaussian elimination
indexMat=np.zeros((nonZeroD+nonZeroR,2))
indexMatD=np.zeros((nonZeroD,2))   
indexMatR=np.zeros((nonZeroR,2))
    

k = 0
d = 0
r = 0
Posd = np.zeros(nonZeroD)
Posr = np.zeros(nonZeroR) 
for i in range(0,nnodes):
    for j in range(0,Range):
        try:
            DispBC_ij=float(Disps[i][j])
            dP=np.dot(DispBC_ij,A[:,k])
            u = np.delete(u, (k), axis=0)
            P -= dP
            A = np.delete(A, (k), axis=0)
            A = np.delete(A, (k), axis=1)
            P = np.delete(P, (k), axis=0)
        except:
            indexMat[k]=[i,j]
            indexMatD[d]=[i,j]
            Posd[d] = k
            k+=1
            d+=1
    for n in range(0,Range):
        try:
            RotBC_ij=float(Rotations[i][n])
            dP=np.dot(RotBC_ij,A[:,k])
            u = np.delete(u, (k), axis=0)
            P -= dP
            A = np.delete(A, (k), axis=0)
            A = np.delete(A, (k), axis=1)
            P = np.delete(P, (k), axis=0)
        except:
            indexMat[k]=[i,n+Range]
            indexMatR[r]=[i,n]
            Posr[r] = k
            k+=1
            r+=1
    

Ainv = np.linalg.inv(A)
u=np.dot(P,Ainv)

for i in range(0,nonZeroD):
    Disps[int(indexMatD[i,0])][int(indexMatD[i,1])]=u[int(Posd[i])] #substitute disp values

for i in range(0,nonZeroR):
    Rotations[int(indexMatR[i,0])][int(indexMatR[i,1])]=u[int(Posr[i])] #substitude rotation values    
    
for i in range(0,nonZeroD+nonZeroR):
    u_0[int(indexMat[i,0])*ndof+int(indexMat[i,1])]=u[i] #linearised displacement vector    
    
#Calculate reaction forces
	
P = np.zeros(ndof*nnodes)
P = np.dot(K,u_0)    
print('displacements:',Disps)
print('rotations:',Rotations)
print('force vector:',P)

u_0=np.reshape(u_0,(nnodes,ndof))
P=np.reshape(P,(nnodes,ndof))
#P_out = np.reshape(P_out,(nnodes,ndof))    
    
#Output Json file
L_out = np.zeros((nnodes,3))
M_out = np.zeros((nnodes,3))
D_out = np.zeros((nnodes,3))
R_out = np.zeros((nnodes,3))
for i in range(0,nnodes):
    for j in range(0,3):
        L_out[i][j] = P[i][j]
        M_out[i][j] = P[i][j+3]
        D_out[i][j] = u_0[i][j]
        R_out[i][j] = u_0[i][j+3] 

j=0
for i in jsonToPython['node']:
    i['Load']=list(L_out[j][:])
    i['Moment']=list(M_out[j][:])
    i['Disp']=list(D_out[j][:])
    i['Rotation']=list(R_out[j][:])
    j+=1
    
outputFile=inputFile+".out"
with open(outputFile,'w') as outfile:
	json.dump(jsonToPython, outfile, indent=4)     
    
    
    