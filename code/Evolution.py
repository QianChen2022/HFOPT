# -*- coding: utf-8 -*-
"""
Holographic First-Order Phase Transition
author: Qian Chen
email: chenqian192@mails.ucas.ac.cn
"""

from __future__ import division 
import numpy as np

def Chebyshev(number,zmin,zmax):
    """
    Function
    ----------
    Chebyshev pseudospectral method

    Parameters
    ----------
    number : int
        the number of Chebyshev points.
        !!! Note : Actual number is number+1 : 0, 1, ……, number
    zmin : int
        the minimum value of the domain.
    zmax : int
        the maximum value of the domain.

    Returns
    -------
    result1 : numpy.array
        ${\rm cos}(j\pi/N),\quad j=0,1,……,number$, mapping to the domain.
        !!! Note : there are number+1 Chebyshev points
    result2 : numpy.array
        Chebyshev first-order differentiation matrix.

    """
    L=zmax-zmin;#区间长度
    N=int(number-1);#N+1个Chebyshev points
    #初始化
    D1=np.zeros((N+1,N+1));
    #计算Chebyshev points
    z=np.cos(np.arange(0,N+1,dtype=int)*np.pi/N);
    #计算 first order Chebyshev Differentiation matrix
    c=np.ones(N+1);c[0]=2;c[-1]=2;
    for i in np.arange(0,N+1,dtype=int):
        for j in np.arange(0,N+1,dtype=int):
            if i==j:
                continue
            else:
                D1[i,j]=(-1)**(i+j)*c[i]/(c[j]*(z[i]-z[j]));
    D1Rowsum=np.sum(D1,axis=1);
    D1=D1-np.diag(D1Rowsum);
    return L*(z+1)/2+zmin,2*D1/L

def Fourier(number,xmin,period):
    """
    Function
    ----------
    Fourier pseudospectral method
    #period=period*pi

    Parameters
    ----------
    number : int
        the number of Fourier points.
        !!! Note : the number must be even
    xmin : int
        the minimum value of the domain.
    period : int
        the period of the domain, in the other word, the length of the domain.
        !!! Note : Actual domain period is period*pi, so the unit of period is pi

    Returns
    -------
    result1 : numpy.array
        $j\times period\times \pi/number,\quad j=0,……,number-1$, mapping to the domain.
        !!! Note : there are number (even) Fourier points, and include starting point of the domain, no end point of the domain
    result2 : numpy.array
        Fourier first-order differentiation matrix.
    result3 : numpy.array
        Fourier second-order differentiation matrix.
    result4 : numpy.array
        Fourier third-order differentiation matrix.

    """
    N=int(number);
    h=2*np.pi/N;#间隔
    #初始化
    D1=np.zeros((N,N));
    D2=np.zeros((N,N));
    D3=np.zeros((N,N));
    #计算Fourier points
    x=xmin+np.arange(0,N,dtype=int)*(period*np.pi)/N;
    for i in np.arange(1,N,dtype=int):
        for j in np.arange(0,i,dtype=int):
            D1[i,j]=(-1)**(i-j)/(2*np.tan((i-j)*h/2));
            D2[i,j]=-(-1)**(i-j)/(2*(np.sin((i-j)*h/2))**2);
            D3[i,j]=(-1)**(i-j)*(3/(4*(np.sin((i-j)*h/2))**2)-N**2/8)/np.tan((i-j)*h/2);
    D1=D1-D1.T;
    D2=D2+D2.T-((N**2+2)/12)*np.eye(N);
    D3=D3-D3.T;
    return x,(2/period)*D1,((2/period)**2)*D2,((2/period)**3)*D3
    
def Quench(t):
    #Gaussion
    T=10;
    w=0.5;
    H=0.6;
    #quench=H*np.exp(-w*(t-T)**2);
    quench=0;
    pt_quench=-2*w*(t-T)*quench;
    return (pt_quench,quench+1.0)


'''1. Spectrum'''
N=int(60);
M=int(360);
space_length=24;

z,D1=Chebyshev(N,0,1);
D2=np.dot(D1,D1);
x,M1,M2,M3=Fourier(M,0,space_length);
M1T=M1.T;
M2T=M2.T;

'''2. Boundary condition'''
first_condition_horizon=np.insert(np.zeros(N-1),0,1);
first_condition_boundary=np.insert(np.zeros(N-1),N-1,1);

'''3. Operator'''
z2=z**2;
z3=z**3;
z4=z**4;
matrix_z=np.tile(z[:,None],(1,M));
matrix_z2=np.tile(z2[:,None],(1,M));
matrix_z3=np.tile(z3[:,None],(1,M))
matrix_z4=np.tile(z4[:,None],(1,M));

zD1_operator=np.diag(z).dot(D1);
zD2_operator=np.diag(z).dot(D2);
z2D2_operator=np.diag(z2).dot(D2);

pzoperator_inv=np.linalg.inv(zD1_operator+np.eye(N));
P_Sigma_hat_operator=z2D2_operator+6*zD1_operator+6*np.eye(N);
P_F_hat_operator=zD2_operator+4*D1;
P_Sigma_plus_hat_operator=np.vstack((first_condition_horizon,D1[1:]));
P_G_plus_hat_operator=zD1_operator+2*np.eye(N);
P_phi_plus_hat_operator=zD1_operator+np.eye(N);
P_A_hat_operator=np.vstack((first_condition_horizon,(zD2_operator+2*D1)[1:]));

'''4. scalar potential'''
index_01=[i < 0.1 for i in z].index(True);
    

'''5. Evolution'''
def Evolution(shift,f1,phi_G_hat,source):
    phi_hat,G_hat=phi_G_hat;
    Bphi2=phi_hat[-1];
    Bg3=G_hat[-1];
    pt_Bphi1,Bphi1=source;
    
    matrix_shift=np.tile(shift,(N,1));
    matrix_px_shift=np.tile(M1.dot(shift),(N,1));
    matrix_p2x_shift=np.tile(M2.dot(shift),(N,1));
    matrix_Bphi2=np.tile(Bphi2,(N,1));
    matrix_Bg3=np.tile(Bg3,(N,1));
    
    pz_phi_hat=D1.dot(phi_hat);
    p2z_phi_hat=D2.dot(phi_hat);
    px_phi_hat=phi_hat.dot(M1T);
    p2x_phi_hat=phi_hat.dot(M2T);
    pxpz_phi_hat=pz_phi_hat.dot(M1T);
    
    phi=Bphi1*matrix_z+matrix_z2*phi_hat;
    pz_phi=Bphi1+matrix_z2*pz_phi_hat+2*matrix_z*phi_hat;
    p2z_phi=2*phi_hat+4*matrix_z*pz_phi_hat+matrix_z2*p2z_phi_hat;
    px_phi=matrix_z2*px_phi_hat;
    pxpz_phi=2*matrix_z*px_phi_hat+matrix_z2*pxpz_phi_hat;
    
    pz_G_hat=D1.dot(G_hat);
    p2z_G_hat=D2.dot(G_hat);
    px_G_hat=G_hat.dot(M1T);
    p2x_G_hat=G_hat.dot(M2T);
    pxpz_G_hat=pz_G_hat.dot(M1T);
    
    G=1+matrix_z3*G_hat;
    pz_G=matrix_z3*pz_G_hat+3*matrix_z2*G_hat;
    p2z_G=6*matrix_z*G_hat+6*matrix_z2*pz_G_hat+matrix_z3*p2z_G_hat;
    px_G=matrix_z3*px_G_hat;
    pxpz_G=3*matrix_z2*px_G_hat+matrix_z3*pxpz_G_hat;
    
    G_hat_G=G_hat/G;
    pz_G_hat_G=pz_G_hat/G;
    px_G_hat_G=px_G_hat/G;
    p2x_G_hat_G=p2x_G_hat/G;
    pxpz_G_hat_G=pxpz_G_hat/G;
    
    pz_G_G=pz_G/G;
    p2z_G_G=p2z_G/G;
    px_G_G=px_G/G;
    pxpz_G_G=pxpz_G/G;
    
    phi_o1=Bphi1+matrix_z*phi_hat;
    V_cosh_tilde=np.vstack(((np.cosh(phi[0:index_01]/np.sqrt(3))-1-phi[0:index_01]**2/6+phi[0:index_01]**4/30)/matrix_z4[0:index_01],41*phi_o1[index_01:]**4/1080+matrix_z2[index_01:]*phi_o1[index_01:]**6/19440+matrix_z4[index_01:]*phi_o1[index_01:]**8/3265920+matrix_z2[index_01:]*matrix_z4[index_01:]*phi_o1[index_01:]**10/881798400));
    V_sinh_tilde=np.vstack(((np.sqrt(3)*np.sinh(phi[0:index_01]/np.sqrt(3))-phi[0:index_01]+0.4*phi[0:index_01]**3)/(3*matrix_z3[0:index_01]),41*phi_o1[index_01:]**3/270+matrix_z2[index_01:]*phi_o1[index_01:]**5/3240+matrix_z4[index_01:]*phi_o1[index_01:]**7/408240+matrix_z2[index_01:]*matrix_z4[index_01:]*phi_o1[index_01:]**9/88179840));
    
    '''1. Sigma'''
    
    Q_Sigma_hat_z2=pz_phi**2+pz_G_G**2;
    Q_Sigma_hat=0.25*matrix_z2*Q_Sigma_hat_z2;
    S_Sigma_hat=-((0.25*matrix_shift-0.03125*Bphi1**2*matrix_z)*Q_Sigma_hat_z2+0.25*matrix_z3*((matrix_z2*pz_G_hat_G+6*matrix_z*G_hat_G)*pz_G_hat_G+pz_phi_hat**2+9*G_hat_G**2)+matrix_z2*phi_hat*pz_phi_hat+matrix_z*(0.5*Bphi1*pz_phi_hat+phi_hat**2)+Bphi1*phi_hat);
    Sigma_hat=np.zeros((N,M));
    for i in range(0,M):
        Sigma_hat[:,i]=np.linalg.solve(P_Sigma_hat_operator+np.diag(Q_Sigma_hat[:,i]),S_Sigma_hat[:,i]);
    
    pz_Sigma_hat=D1.dot(Sigma_hat);
    px_Sigma_hat=Sigma_hat.dot(M1T);
    p2x_Sigma_hat=Sigma_hat.dot(M2T);
    pxpz_Sigma_hat=pz_Sigma_hat.dot(M1T);
    
    Sigma_tilde=1+matrix_shift*matrix_z-0.125*Bphi1**2*matrix_z2+matrix_z3*Sigma_hat;
    Sigma_tilde2=Sigma_tilde**2;
    G_Sigma_tilde=G*Sigma_tilde;
    G_Sigma_tilde2=G*Sigma_tilde2;
    pz_Sigma_tilde=matrix_shift-0.25*Bphi1**2*matrix_z+3*matrix_z2*Sigma_hat+matrix_z3*pz_Sigma_hat;
    px_Sigma_tilde=matrix_z*matrix_px_shift+matrix_z3*px_Sigma_hat;
    pxpz_Sigma_tilde=matrix_px_shift+3*matrix_z2*px_Sigma_hat+matrix_z3*pxpz_Sigma_hat;
    
    Sigma_hat_Sigma_tilde=Sigma_hat/Sigma_tilde;
    pz_Sigma_hat_Sigma_tilde=pz_Sigma_hat/Sigma_tilde;
    px_Sigma_hat_Sigma_tilde=px_Sigma_hat/Sigma_tilde;
    p2x_Sigma_hat_Sigma_tilde=p2x_Sigma_hat/Sigma_tilde;
    pxpz_Sigma_hat_Sigma_tilde=pxpz_Sigma_hat/Sigma_tilde;
    
    pz_Sigma_tilde_Sigma_tilde=pz_Sigma_tilde/Sigma_tilde;
    px_Sigma_tilde_Sigma_tilde=px_Sigma_tilde/Sigma_tilde;
    pxpz_Sigma_tilde_Sigma_tilde=pxpz_Sigma_tilde/Sigma_tilde;
    
    
    '''2. F'''
    Q_F_hat_z1=1.5*pz_G_G**2+0.5*pz_phi**2-2*pz_Sigma_tilde_Sigma_tilde*(pz_Sigma_tilde_Sigma_tilde+pz_G_G)-p2z_G_G;
    P1_F_hat=-matrix_z*pz_G_G;
    Q_F_hat=matrix_z*Q_F_hat_z1+4*pz_Sigma_tilde_Sigma_tilde-pz_G_G;
    S_F_hat=-(-Q_F_hat_z1*matrix_px_shift\
              +matrix_z3*(2*(matrix_z*pz_G_hat_G+3*G_hat_G)*(px_Sigma_hat_Sigma_tilde-px_G_hat_G)+2*matrix_z*(matrix_px_shift*Sigma_hat_Sigma_tilde**2+px_Sigma_hat_Sigma_tilde*pz_Sigma_hat_Sigma_tilde)-0.5*Bphi1**2*matrix_px_shift*Sigma_hat_Sigma_tilde/Sigma_tilde+6*Sigma_hat_Sigma_tilde*px_Sigma_hat_Sigma_tilde)\
                  +matrix_z2*((2*matrix_px_shift*(pz_G_hat_G+pz_Sigma_hat_Sigma_tilde)-0.5*Bphi1**2*px_Sigma_hat_Sigma_tilde)/Sigma_tilde+0.03125*Bphi1**4*matrix_px_shift/Sigma_tilde2-4*matrix_px_shift*pz_Sigma_hat_Sigma_tilde-px_phi_hat*pz_phi_hat)\
                      +2*matrix_z*((matrix_shift*px_Sigma_hat_Sigma_tilde+matrix_px_shift*(4*Sigma_hat_Sigma_tilde+3*G_hat_G))/Sigma_tilde-6*matrix_px_shift*Sigma_hat_Sigma_tilde-pxpz_Sigma_hat_Sigma_tilde-phi_hat*px_phi_hat+0.5*pxpz_G_hat_G)\
                          +Bphi1**2*matrix_px_shift/Sigma_tilde-(2*matrix_shift**2+0.75*Bphi1**2)*matrix_px_shift/Sigma_tilde2-Bphi1*px_phi_hat+3*px_G_hat_G-6*px_Sigma_hat_Sigma_tilde);
    replace_S_F_hat=np.vstack((S_F_hat[:-1],f1));
    F_hat=np.zeros((N,M));
    for i in range(0,M):
        F_hat[:,i]=np.linalg.solve(np.vstack(((P_F_hat_operator+np.diag(P1_F_hat[:,i]).dot(D1)+np.diag(Q_F_hat[:,i]))[:-1],first_condition_boundary)),replace_S_F_hat[:,i]);
        
    pz_F_hat=D1.dot(F_hat);
    #p2z_F_hat=D2.dot(F_hat);
    px_F_hat=F_hat.dot(M1T);
    pxpz_F_hat=pz_F_hat.dot(M1T);
    
    F=-matrix_px_shift+matrix_z*F_hat;
    F2=F**2;
    pz_F=matrix_z*pz_F_hat+F_hat;
    #p2z_F=2*pz_F_hat+matrix_z*p2z_F_hat;
    p2z_F=S_F_hat-Q_F_hat*F_hat-(2+P1_F_hat)*pz_F_hat;
    px_F=-matrix_p2x_shift+matrix_z*px_F_hat;
    pxpz_F=matrix_z*pxpz_F_hat+px_F_hat;
    F_pz_F=F*pz_F;
    
    
    '''3. Sigma_plus'''
    S_Sigma_plus_hat=-(0.5*(matrix_z2*(0.25*F2*pz_phi**2+F_pz_F*(0.5*pz_G_G-pz_Sigma_tilde_Sigma_tilde)-0.25*pz_F**2)+matrix_z*F_pz_F+px_F*(pz_Sigma_tilde_Sigma_tilde-0.5*pz_G_G)-0.5*pz_F*px_G_G+0.5*pxpz_F+(pxpz_Sigma_tilde_Sigma_tilde-0.5*pxpz_G_G+px_G_G*pz_G_G-px_Sigma_tilde_Sigma_tilde*pz_Sigma_tilde_Sigma_tilde-px_G_G*pz_Sigma_tilde_Sigma_tilde)*F)/G+1.5*V_cosh_tilde*Sigma_tilde2\
                       +0.5*matrix_z4*((px_Sigma_hat_Sigma_tilde**2+px_G_hat_G*px_Sigma_hat_Sigma_tilde-px_G_hat_G**2)/G+(Bphi1*matrix_z*phi_hat+0.5*Bphi1**2)*Sigma_hat**2-0.25*Bphi1**3*Sigma_hat*phi_hat)\
                           +matrix_z3*((matrix_shift*Bphi1*phi_hat-Bphi1**4/16)*Sigma_hat+Bphi1**5*phi_hat/128+0.5*F_hat*px_G_hat_G/G)\
                               +0.5*matrix_z2*((matrix_p2x_shift*Sigma_hat_Sigma_tilde+matrix_px_shift*(px_G_hat_G+2*px_Sigma_hat_Sigma_tilde)/Sigma_tilde-0.25*px_phi_hat**2-matrix_px_shift*px_G_hat_G)/G+(matrix_shift**2-Bphi1**2/8)*pz_Sigma_hat+3*Sigma_hat**2+(matrix_shift*Bphi1**2+2*Bphi1*phi_hat)*Sigma_hat+Bphi1**6/128-0.25*matrix_shift*Bphi1**3*phi_hat)\
                                   +matrix_z*(0.5*(0.5*p2x_G_hat_G-p2x_Sigma_hat_Sigma_tilde-Bphi1**2*matrix_p2x_shift/(8*Sigma_tilde))/G+matrix_shift**2*Sigma_hat-matrix_shift*Bphi1**4/16+matrix_shift*pz_Sigma_hat+(0.5*matrix_shift**2*Bphi1-Bphi1**3/8)*phi_hat)\
                                       +0.5*(matrix_shift*matrix_p2x_shift/Sigma_tilde+matrix_px_shift**2/Sigma_tilde2-px_F_hat)/G+3*matrix_shift**2*Bphi1**2/16+4*matrix_shift*Sigma_hat+matrix_shift*Bphi1*phi_hat-Bphi1**4/32+0.5*pz_Sigma_hat+0.25*Sigma_tilde2*phi_hat**2+pzoperator_inv.dot(3*pz_Sigma_hat+0.5*Bphi1*pz_phi_hat));
    
    H_Sigma_plus_hat=(F[0]**2*(pz_Sigma_tilde_Sigma_tilde[0]-1)-px_F[0]+F[0]*px_G_G[0])/(2*G_Sigma_tilde[0])-0.5*shift**2-shift+Bphi1**2/16-0.5;
    Sigma_plus_hat=np.linalg.solve(P_Sigma_plus_hat_operator,np.vstack((Sigma_tilde[0]*H_Sigma_plus_hat,S_Sigma_plus_hat[1:])))/Sigma_tilde;
    
    
    '''4. G_plus'''
    Q_G_plus_hat=matrix_z*(pz_Sigma_tilde_Sigma_tilde-pz_G_G);
    S_G_plus_hat=-(Sigma_plus_hat*pz_G/Sigma_tilde+1.5*matrix_Bg3*(pz_G_G-pz_Sigma_tilde_Sigma_tilde)\
                   +(matrix_z2*((pz_Sigma_tilde_Sigma_tilde-0.5*pz_G_G)*F_pz_F+(0.75*pz_G_G**2-pz_Sigma_tilde_Sigma_tilde**2-0.5*p2z_G_G-pz_G_G*pz_Sigma_tilde_Sigma_tilde)*F2-0.25*pz_F**2)+matrix_z*(2*F2*pz_Sigma_tilde_Sigma_tilde-F_pz_F)+(0.5*pxpz_G_G-pxpz_Sigma_tilde_Sigma_tilde+px_Sigma_tilde_Sigma_tilde*pz_Sigma_tilde_Sigma_tilde+pz_G_G*px_Sigma_tilde_Sigma_tilde-px_G_G*pz_G_G)*F-F2+0.5*pxpz_F-pz_F*px_Sigma_tilde_Sigma_tilde)/Sigma_tilde2\
                       +(matrix_z2*(0.5*matrix_shift**2*pz_G_hat-Bphi1**2*pz_G_hat/16-0.25*px_phi_hat**2/Sigma_tilde)+matrix_z*(1.5*matrix_shift**2*G_hat-3*Bphi1**2*G_hat/16+matrix_shift*pz_G_hat)+3*matrix_shift*G_hat+0.5*pz_G_hat)/Sigma_tilde+1.5*(pzoperator_inv.dot(pz_G_hat-matrix_Bg3*pz_Sigma_tilde))/Sigma_tilde);
    G_plus_hat=np.zeros((N,M));
    for i in range(0,M):
        G_plus_hat[:,i]=np.linalg.solve(P_G_plus_hat_operator+np.diag(Q_G_plus_hat[:,i]),S_G_plus_hat[:,i]);
    
    
    '''5. phi_plus'''
    Q_phi_plus_hat=matrix_z*pz_Sigma_tilde_Sigma_tilde;
    S_phi_plus_hat=-((matrix_z3*(0.5*(pz_G_G*pz_phi-p2z_phi)*F2-F_pz_F*pz_phi)-matrix_z2*F2*pz_phi+0.5*matrix_z*((2*pxpz_phi-px_G_G*pz_phi-pz_G_G*px_phi)*F+px_F*pz_phi+pz_F*px_phi))/G_Sigma_tilde\
                     +(matrix_z*Sigma_plus_hat+0.5*matrix_shift**2-Bphi1**2/16)*pz_phi-(matrix_shift*Bphi1+matrix_Bphi2-pt_Bphi1)*pz_Sigma_tilde-3*Sigma_tilde*V_sinh_tilde\
                         +matrix_z2*(0.5*matrix_z2*px_phi_hat*px_G_hat_G/G_Sigma_tilde-Sigma_hat*phi_hat-0.5*Bphi1*pz_Sigma_hat)+matrix_z*(matrix_shift*pz_phi_hat-2*Bphi1*Sigma_hat+0.125*Bphi1**2*phi_hat-0.5*p2x_phi_hat/G_Sigma_tilde)+matrix_shift*phi_hat+3.0*Bphi1**3/16+0.5*pz_phi_hat)/Sigma_tilde;
    phi_plus_hat=np.zeros((N,M));
    for i in range(0,M):
        phi_plus_hat[:,i]=np.linalg.solve(P_phi_plus_hat_operator+np.diag(Q_phi_plus_hat[:,i]),S_phi_plus_hat[:,i]);
    
    
    '''6. A'''
    S_A_hat=-(0.5*matrix_z3*(F*(p2z_F-pz_F*pz_G_G)+pz_F**2)/G_Sigma_tilde2+matrix_z2*(F_pz_F/G_Sigma_tilde2-0.5*G_plus_hat*pz_G_G/G)\
              +matrix_z*(0.5*(pz_F*px_G_G-pxpz_F)/G_Sigma_tilde2-3*V_cosh_tilde-0.5*phi_plus_hat*pz_phi-2*D1.dot(Sigma_plus_hat)/Sigma_tilde+0.75*matrix_Bg3*pz_G_G/G)\
                  +0.5*(matrix_shift*Bphi1+matrix_Bphi2-pt_Bphi1)*pz_phi-2*Sigma_plus_hat/Sigma_tilde-0.25*Bphi1**2*matrix_z2*Sigma_hat_Sigma_tilde+matrix_z*(Bphi1**4/(32*Sigma_tilde)+0.25*Bphi1*pz_phi_hat-0.5*phi_hat**2)-0.25*Bphi1**2*matrix_shift/Sigma_tilde-2*Sigma_hat_Sigma_tilde-0.5*Bphi1*phi_hat);
    
    H_pz_Sigma_Sigma=pz_Sigma_tilde_Sigma_tilde[0]-1;
    H_Sigma_plus=0.5*(1+shift)**2-Bphi1**2/16+Sigma_plus_hat[0];
    H_pz_Sigma_plus=-(1+shift)+Sigma_plus_hat[0]+D1.dot(Sigma_plus_hat)[0];
    H_G_plus=-1.5*G_hat[-1]+G_plus_hat[0];
    H_phi_plus=-0.5*Bphi1-(shift*Bphi1+phi_hat[-1]-pt_Bphi1)+phi_plus_hat[0];
    
    H_Px_A=pz_F[0]-F[0]*(2*H_pz_Sigma_Sigma+pz_G_G[0])-px_G_G[0];
    H_Q_A=(H_pz_Sigma_Sigma**2+pz_G_G[0]*H_pz_Sigma_Sigma+0.25*pz_G_G[0]**2)*F2[0]-(0.5*pz_G_G[0]+H_pz_Sigma_Sigma)*F_pz_F[0]\
        +0.5*(0.5*pz_F[0]-px_G_G[0])*pz_F[0]-(0.5*pz_G_G[0]+H_pz_Sigma_Sigma)*px_F[0]+0.5*pxpz_F[0]+(H_pz_Sigma_Sigma*px_G_G[0]+px_Sigma_tilde_Sigma_tilde[0]*H_pz_Sigma_Sigma+px_G_G[0]*pz_G_G[0]-0.5*pxpz_G_G[0]-(pxpz_Sigma_tilde_Sigma_tilde[0]-px_Sigma_tilde_Sigma_tilde[0]))*F[0]\
            +0.25*px_phi[0]**2+(M2.dot(shift)+p2x_Sigma_hat[0])/Sigma_tilde[0]-px_Sigma_tilde_Sigma_tilde[0]**2+px_G_G[0]**2-0.5*p2x_G_hat_G[0]-px_Sigma_tilde_Sigma_tilde[0]*px_G_G[0]+0.5*G[0]*(-6*np.cosh(phi[0]/np.sqrt(3))-0.2*phi[0]**4)*Sigma_tilde[0]**2;
    H_S_A=(0.5*pz_phi[0]*H_phi_plus+0.5*pz_G_G[0]*H_G_plus/G[0]+H_pz_Sigma_Sigma*H_Sigma_plus/Sigma_tilde[0]+H_pz_Sigma_plus/Sigma_tilde[0])*F2[0]+(2*H_G_plus*px_Sigma_tilde_Sigma_tilde[0]/G[0]+2*px_G_G[0]*H_Sigma_plus/Sigma_tilde[0]-H_phi_plus*px_phi[0])*F[0]-(2*H_Sigma_plus/Sigma_tilde[0]+H_G_plus/G[0])*px_F[0]\
        +(F2[0]**2*H_pz_Sigma_Sigma*(0.5*pz_G_G[0]+H_pz_Sigma_Sigma)+F[0]**3*((px_G_G[0]-pz_F[0]+2*px_Sigma_tilde_Sigma_tilde[0])*H_pz_Sigma_Sigma+0.5*pz_phi[0]*px_phi[0]-0.5*pxpz_G_G[0]+1.5*pz_G_G[0]*px_G_G[0])+F2[0]*(0.5*pxpz_F[0]-0.5*px_phi[0]**2-pz_F[0]*(px_G_G[0]+px_Sigma_tilde_Sigma_tilde[0])-px_F[0]*(pz_G_G[0]+2*H_pz_Sigma_Sigma)-0.5*p2x_G_hat_G[0]-(M2.dot(shift)+p2x_Sigma_hat[0])/Sigma_tilde[0]+px_G_G[0]**2+px_Sigma_tilde_Sigma_tilde[0]**2+3*px_G_G[0]*px_Sigma_tilde_Sigma_tilde[0])+F[0]*(pz_F[0]*px_F[0]+(-M3.dot(shift)+M2.dot(F_hat[0]))-2*px_F[0]*px_G_G[0]-2*px_F[0]*px_Sigma_tilde_Sigma_tilde[0]))/G_Sigma_tilde2[0]\
            +(-0.5*G[0]*H_phi_plus**2-0.5*H_G_plus**2/G[0])*Sigma_tilde2[0]-2*G[0]*H_Sigma_plus**2;
    H_A=np.linalg.solve(M2+np.diag(H_Px_A).dot(M1)+np.diag(H_Q_A),-H_S_A);
    A_hat=np.linalg.solve(P_A_hat_operator,np.vstack((H_A-0.5*(1+shift)**2+Bphi1**2/8,S_A_hat[1:])));
    
    '''9. Evolution '''
    pt_phi_hat=matrix_z2*(0.5*matrix_shift**2-Bphi1**2/8+A_hat)*pz_phi_hat+matrix_z*((matrix_shift**2-Bphi1**2/4+2*A_hat)*phi_hat+matrix_shift*pz_phi_hat)+0.5*Bphi1*matrix_shift**2+2*matrix_shift*phi_hat-Bphi1**3/8+Bphi1*A_hat+phi_plus_hat+0.5*pz_phi_hat+pzoperator_inv.dot(pz_phi_hat);
    pt_G_hat=matrix_z2*(0.5*matrix_shift**2-Bphi1**2/8+A_hat)*pz_G_hat+matrix_z*((1.5*matrix_shift**2-3*Bphi1**2/8+3*A_hat)*G_hat+matrix_shift*pz_G_hat)+3*matrix_shift*G_hat+G_plus_hat+0.5*pz_G_hat+1.5*pzoperator_inv.dot(pz_G_hat);
    pt_shift=-A_hat[-1];
    pt_f1=2*M1.dot(Sigma_plus_hat[-1])/3-Bphi1**2*M1.dot(shift)/9-Bphi1*M1.dot(phi_hat[-1])/9-M1.dot(G_hat[-1]);
    
    return pt_shift,pt_f1,np.array([pt_phi_hat,pt_G_hat]),np.array([Sigma_hat,F_hat,Sigma_plus_hat,G_plus_hat,phi_plus_hat,A_hat]),np.array([0.5*(Bphi1*shift+phi_hat[-1]-pt_Bphi1),-2*Sigma_plus_hat[-1]-shift*Bphi1**2/6-Bphi1*phi_hat[-1]/6,-1.5*F_hat[-1],shift*Bphi1**2/6+Bphi1*phi_hat[-1]/6-Bphi1*pt_Bphi1/4-Sigma_plus_hat[-1]+1.5*G_hat[-1],shift*Bphi1**2/6+Bphi1*phi_hat[-1]/6-Bphi1*pt_Bphi1/4-Sigma_plus_hat[-1]-1.5*G_hat[-1]])

def RK4_Evolution(dt,shift,f1,phi_G_hat,t):
    K1_shift,K1_f1,K1_phi_G_hat,data_fields,physics=Evolution(shift,f1,phi_G_hat,Quench(t));
    K2_shift,K2_f1,K2_phi_G_hat,fie_useless,useless=Evolution(shift+K1_shift*(dt/2),f1+K1_f1*(dt/2),phi_G_hat+K1_phi_G_hat*(dt/2),Quench(t+dt/2));
    K3_shift,K3_f1,K3_phi_G_hat,fie_useless,useless=Evolution(shift+K2_shift*(dt/2),f1+K2_f1*(dt/2),phi_G_hat+K2_phi_G_hat*(dt/2),Quench(t+dt/2));
    K4_shift,K4_f1,K4_phi_G_hat,fie_useless,useless=Evolution(shift+K3_shift*dt,f1+K3_f1*dt,phi_G_hat+K3_phi_G_hat*dt,Quench(t+dt));
    return shift+(K1_shift+2*K2_shift+2*K3_shift+K4_shift)*(dt/6),f1+(K1_f1+2*K2_f1+2*K3_f1+K4_f1)*(dt/6),phi_G_hat+(K1_phi_G_hat+2*K2_phi_G_hat+2*K3_phi_G_hat+K4_phi_G_hat)*(dt/6),data_fields,physics

def Error(dt,shift,f1,phi_G_hat,t):
    shift_next,f1_next,phi_G_hat_next,data_fields1,physics1=RK4_Evolution(dt,shift,f1,phi_G_hat,t);
    shift_next,f1_next,phi_G_hat_next,data_fields2,physics2=RK4_Evolution(dt,shift_next,f1_next,phi_G_hat_next,t+dt);
    shift_next,f1_next,phi_G_hat_next,data_fields3,physics3=RK4_Evolution(dt,shift_next,f1_next,phi_G_hat_next,t+2*dt);
    shift_next,f1_next,phi_G_hat_next,data_fields4,physics4=RK4_Evolution(dt,shift_next,f1_next,phi_G_hat_next,t+3*dt);
    shift_next,f1_next,phi_G_hat_next,data_fields5,physics5=RK4_Evolution(dt,shift_next,f1_next,phi_G_hat_next,t+4*dt);
    pt_T00=M1.dot(physics3[2])-physics3[0]*Quench(t+2*dt)[0];
    error=(physics1[1]-8*physics2[1]+8*physics4[1]-physics5[1])/(12*dt)-pt_T00;
    return shift_next,f1_next,phi_G_hat_next,np.array([pt_T00,error])
    
    