# -*- coding: utf-8 -*-
"""
Holographic First-Order Phase Transition
author: Qian Chen
email: chenqian192@mails.ucas.ac.cn
"""

from __future__ import division 
import os
import numpy as np
import time
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import Evolution as es
'''
1. Spectrum
'''
dt=0.005;
dataspace=int(200);
tpointcount=int(400);
datacount=tpointcount+1;

N=es.N;
M=es.M;
z,x,D1,pzoperator_inv=es.z,es.x,es.D1,es.pzoperator_inv;

'''
2. Initial value
'''
file_location=r'..';

shift=np.zeros((datacount,M));
f1=np.zeros((datacount,M));
phi_G_hat=np.zeros((datacount,2,N,M));
data_fields=np.zeros((datacount,6,N,M));
physics=np.zeros((datacount,5,M));
error=np.zeros((datacount,2,M));

initial_data=np.load(file_location+r'\1.initial_data\initial_data.npy');
shift[0]=initial_data[2,-1]*np.ones(M,dtype=np.float64);
f1[0]=np.zeros(M,dtype=np.float64);
initial_phi_hat=(np.kron(pzoperator_inv.dot(D1.dot(initial_data[0])),np.ones(M,dtype=np.float64))-0.1*np.kron((1-z)**2,np.exp(-10*(np.cos(x/24))**2))).reshape(N,M);
init_G_hat=np.zeros((N,M));
phi_G_hat[0]=np.array([initial_phi_hat,init_G_hat]);

'''
3. Evolution
'''
time_start=time.time();
for i in range(0,datacount-1):
    print(i);
    shift_next,f1_next,phi_G_hat_next,data_fields[i],physics[i]=es.RK4_Evolution(dt,shift[i],f1[i],phi_G_hat[i],i*dataspace*dt);
    shift_next,f1_next,phi_G_hat_next,error[i]=es.Error(dt,shift_next,f1_next,phi_G_hat_next,(i*dataspace+1)*dt);
    for j in range(6,dataspace):
        shift_next,f1_next,phi_G_hat_next,data_fields_j,physics_j=es.RK4_Evolution(dt,shift_next,f1_next,phi_G_hat_next,(j+i*dataspace)*dt);
    shift[i+1]=shift_next.copy();
    f1[i+1]=f1_next.copy();
    phi_G_hat[i+1]=phi_G_hat_next.copy();
    print((time.time()-time_start)/3600);
shift_next,f1_next,phi_G_hat_next,data_fields[datacount-1],physics[datacount-1]=es.RK4_Evolution(dt,shift[datacount-1],f1[datacount-1],phi_G_hat[datacount-1],(datacount-1)*dataspace*dt);
shift_next,f1_next,phi_G_hat_next,error[datacount-1]=es.Error(dt,shift_next,f1_next,phi_G_hat_next,((datacount-1)*dataspace+1)*dt);

'''
4. Save data
'''
os.makedirs(file_location+r'\2.data', exist_ok=True)
np.save(file_location+r'\2.data\1.shift.npy',shift);
np.save(file_location+r'\2.data\2.f1.npy',f1);
np.save(file_location+r'\2.data\3.phi_G_hat.npy',phi_G_hat);
np.save(file_location+r'\2.data\4.data_fields.npy',data_fields);
np.save(file_location+r'\2.data\5.physics.npy',physics);
np.save(file_location+r'\2.data\6.error.npy',error);

'''
5. data visualization
'''
os.makedirs(file_location+r'\3.picture', exist_ok=True)
entropy_ave=np.zeros(datacount);
energy_ave=np.zeros(datacount);
Bphi1=np.zeros(datacount);
for i in range(0,datacount):
    Bphi1[i]=(-physics[i,1,0]+physics[i,3,0]+physics[i,4,0])/physics[i,0,0];
    entropy_ave[i]=2*np.pi*np.sum((1+shift[i]-Bphi1[i]**2/8+data_fields[i,0,0])**2)/M;
    energy_ave[i]=np.sum(physics[i,1])/M;

plt.plot(range(0,datacount),Bphi1);
plt.xlabel('t');
plt.ylabel(r'$\phi_{1}$');
plt.savefig(file_location+r'\3.picture\1.phi1.jpeg');
plt.close();

plt.plot(range(0,datacount),energy_ave);
plt.xlabel('t');
plt.ylabel(r'$\bar{E}$');
plt.savefig(file_location+r'\3.picture\2.energy_ave.jpeg');
plt.close();

plt.plot(range(0,datacount),entropy_ave);
plt.xlabel('t');
plt.ylabel(r'$\bar{S}$');
plt.savefig(file_location+r'\3.picture\3.entropy_ave.jpeg');
plt.close();

fig = plt.figure();
ax = Axes3D(fig);
T, X = np.meshgrid(np.arange(0,datacount), x);
ax.plot_surface(T.T, X.T, physics[:,0], rstride=1, cstride=1, cmap='rainbow');
ax.set_xlabel('t');
ax.set_ylabel('x');
ax.set_zlabel(r'$<O_{\phi}>$');
plt.savefig(file_location+r'\3.picture\4.one_point.jpg');
plt.close();

fig = plt.figure();
ax = Axes3D(fig);
T, X = np.meshgrid(np.arange(0,datacount), x);
ax.plot_surface(T.T, X.T, physics[:,1], rstride=1, cstride=1, cmap='rainbow');
ax.set_xlabel('t');
ax.set_ylabel('x');
ax.set_zlabel(r'$\epsilon$');
plt.savefig(file_location+r'\3.picture\5.energy_density.jpg');
plt.close();

fig = plt.figure();
ax = Axes3D(fig);
T, X = np.meshgrid(np.arange(0,datacount), x);
ax.plot_surface(T.T, X.T, physics[:,2], rstride=1, cstride=1, cmap='rainbow');
ax.set_xlabel('t');
ax.set_ylabel('x');
ax.set_zlabel(r'$T_{01}$');
plt.savefig(file_location+r'\3.picture\6.momentum.jpg');
plt.close();

fig = plt.figure();
ax = Axes3D(fig);
T, X = np.meshgrid(np.arange(0,datacount), x);
ax.plot_surface(T.T, X.T, physics[:,3], rstride=1, cstride=1, cmap='rainbow');
ax.set_xlabel('t');
ax.set_ylabel('x');
ax.set_zlabel(r'$T_{11}$');
plt.savefig(file_location+r'\3.picture\7.T11.jpg');
plt.close();

fig = plt.figure();
ax = Axes3D(fig);
T, X = np.meshgrid(np.arange(0,datacount), x);
ax.plot_surface(T.T, X.T, physics[:,4], rstride=1, cstride=1, cmap='rainbow');
ax.set_xlabel('t');
ax.set_ylabel('x');
ax.set_zlabel(r'$T_{22}$');
plt.savefig(file_location+r'\3.picture\8.T22.jpg');
plt.close();

fig = plt.figure();
ax = Axes3D(fig);
T, X = np.meshgrid(np.arange(0,datacount), x);
ax.plot_surface(T.T, X.T, error[:,1], rstride=1, cstride=1, cmap='rainbow');
ax.set_xlabel('t');
ax.set_ylabel('x');
ax.set_zlabel(r'error');
plt.savefig(file_location+r'\3.picture\9.error.jpg');
plt.close();


