import numpy as np
import pandas as pd
from scipy import interpolate
import warnings
warnings.filterwarnings( "ignore")
import matplotlib.pyplot as plt

def gridSPH(m , hs):
    # golden angle
    ga = ((np.sqrt(5)+1)/2)*np.pi;

    if hs == 1:
        t = np.arange(-m, 1)
    elif hs == 2:
        t = np.arange(0, m+1)
    else:
        t = np.arange(-m, m+1)

    theta = np.remainder(2*t*ga, 2*np.pi);
    phi = np.pi/2 - np.arcsin(2*t/(2*m+1));

    x = np.sin(phi)*np.cos(theta);
    y = np.sin(phi)*np.sin(theta);
    z = np.cos(phi);

    return np.vstack((x,y,z))

def ske(X, N_0, k=1):
    k = np.atleast_1d(k)
    C3= k/(4*np.pi*np.sinh(k));
    inner_prods = X.T @ N_0
    tensor_dots = np.tensordot(k, inner_prods, axes = 0)
    sums = np.sum(np.exp(tensor_dots), axis = 2)
    fhat_vMF = (C3.reshape(-1,1)*sums)/N_0.shape[1]
    return fhat_vMF

def GskeSD(X, N_0, k):
    # Assume Xs have norm 1
    k = np.atleast_1d(k)
    C3=k/(4*np.pi*np.sinh(k));
    #norms = np.linalg.norm(X, axis=0)
    inner_prods = X.T @ N_0

    g = np.exp(np.tensordot(k, inner_prods, axes = 0))

    dgx = np.column_stack((X[1]**2 + X[2]**2, -X[0]*X[1], -X[0]*X[2])) @ N_0
    #dgx = dgx/norms.reshape(-1,1)
    dgx = np.sum(dgx*g, axis=2)

    dgy = np.column_stack((-X[1]*X[0], X[0]**2 + X[2]**2, -X[1]*X[2])) @ N_0
    #dgy = dgy/norms.reshape(-1,1)
    dgy = np.sum(dgy*g, axis=2)

    dgz = np.column_stack((-X[2]*X[0], -X[2]*X[1], X[0]**2 + X[1]**2)) @ N_0
    #dgz = dgz/norms.reshape(-1,1)
    dgz = np.sum(dgz*g, axis=2)

    nabla_g = ((C3*k).reshape(-1,1,1) * np.stack((dgx,dgy,dgz), axis=1))/N_0.shape[1]


    sdx = np.std(dgx, ddof=1, axis=1)
    sdy = np.std(dgy, ddof=1, axis=1)
    sdz = np.std(dgz, ddof=1, axis=1)

    SDhat = C3*k*np.asarray((sdx, sdy, sdz))/np.sqrt(N_0.shape[1])

    return nabla_g, SDhat

def BSquantile( N_0, b=50, sph=2, k0 = None, X0 = None):
    #GRID FOR SPHERE
    if X0 is None:
        m = 500
        if sph == 1:
            if np.sum(np.sign(N_0[2])) < 0:
                X = gridSPH(m,1);
            else:
                X = gridSPH(m,2);

        elif sph==2:
            X=gridSPH((m-1)/2,3);
    else:
        m = 1;
        X = X0.copy();

    #BOOTSTRAP SAMPLE AND QUANTILE
    nx = N_0.shape[1];
    if k0 is None:
        nh = 25;
        h = np.linspace(5,100,nh);
    else:
        k0 = np.atleast_1d(k0)
        nh = len(k0);
        h = k0;

    rng = np.random.default_rng()
    Bi = rng.integers(low=0, high=nx, size=(b, nx))
    maxsG = np.zeros(b)

    inner_prods = X.T @ N_0
    ESS = np.sum(np.exp(h[-1]*inner_prods), axis = 1)/np.exp(h[-1])
    mask = ESS > 5
    #ESS = np.empty((nh, X.shape[1]))
    #for i in range(nh):
        #ESS[i] = np.sum(np.exp(h[i]*inner_prods), axis = 1)/np.exp(h[i])
    Dg, _ = GskeSD(X, N_0, h)
    X1 = X[:, mask]

    for i in range(b):
        #for j in range(nh):
        #    mask = ESS[j] > 5
        #    if np.sum(mask) > 0:
        #        X1 = X[:, mask]
        DgB, seDgB = GskeSD(X1, N_0[:, Bi[i]], h)
        Zbxk = np.sum(((DgB - Dg[:, :, mask])/((seDgB.T)[..., np.newaxis]))**2, axis =1)
        maxsG[i] = np.max(Zbxk)

    q = np.sort(maxsG)
    q = q[int(.95*b)]

    return q

def lambPROJ(X, i=None):
    if i is None:
        rho = np.sqrt((2*(1-np.abs(X[2])))/(1-X[2]**2))
    elif i == 1:
        rho = np.sqrt(2/(1+X[2]))
    elif i == 2:
        rho = np.sqrt(2/(1-X[2]))

    rhox = rho*X[0]
    rhoy = rho*X[1]

    return np.vstack((rhox, rhoy))

def contour_hemisphere(X, fhat, X_0, nabla_g, qind, hemisphere = 1):
    if hemisphere == 1 :
        pshemi_mask = X[2] > -0.05
        hemi_mask = X_0[2] > 0
    else:
        pshemi_mask = X[2] < 0.05
        hemi_mask = X_0[2] < 0

    PsHemi = X[:, pshemi_mask]
    Z = fhat[pshemi_mask]
    pslamb = lambPROJ(PsHemi,hemisphere)

    mx = np.linspace(-np.sqrt(2), np.sqrt(2), 120)
    Lgrid = np.asarray(np.meshgrid(mx, mx))
    radii_mask = np.sum(Lgrid**2, axis=0) < 2

    itrplt = interpolate.griddata(points = tuple(pslamb),
                              values = Z,
                              xi = tuple(Lgrid[:, radii_mask]),
                              method='linear', fill_value=0)
    Zq = np.zeros_like(Lgrid[0])
    Zq[radii_mask] = itrplt
    Zq = Zq.reshape(Lgrid[0].shape)

    #qind = np.sum((nabla_g/SDhat.reshape(-1,1))**2, axis=0) > q

    signif_mask = hemi_mask & qind
    Hemi = X_0[:, signif_mask]
    lamb = lambPROJ(X_0[:, signif_mask], hemisphere)
    Dg = nabla_g[:, signif_mask]

    #radii = np.sqrt(np.sum((Hemi + Dg)**2, axis=0))
    #arrowhead = (Hemi + Dg)/radii
    radii = np.sqrt(np.sum((Dg)**2, axis=0))
    arrowhead = (Dg)/radii
    arrlamb = lambPROJ(arrowhead,1)

    return mx, Zq, lamb, arrlamb

def lambert_plot_Z(k, fs = 18, lvls = 12, writefig=False, filename='file'):

    fig, ax = plt.subplots(1, 3, figsize=(15,8), gridspec_kw={"width_ratios":[1,1, 0.05]})

    for i in [0,1]:
        ax[i].set_aspect('equal')
        ax[i].plot(np.sqrt(2)*np.cos(theta), np.sqrt(2)*np.sin(theta), c='silver', lw=2);
        ax[i].tick_params(labelsize=fs)
        ax[i].set_xticks(ticks = [0])
        ax[i].set_yticks(ticks = [0])
        ax[i].set_xticklabels(labels = ['270'])

        ax_t = ax[i].secondary_xaxis('top')
        ax_t.tick_params(axis='x', direction='inout', labelsize=fs)
        ax_t.set_xticks(ticks = [0])
        ax_t.set_xticklabels(labels = ['90'])

    i = 0
    cs = ax[i].contourf(Nmx,Nmx, NZq, levels=lvls, vmax=fhat.max())
    Qq = ax[i].quiver(*Nlamb, *Narrlamb, color='r', scale_units='width')#, scale=10)
    Qq._init()
    assert isinstance(Qq.scale, float)
    ax[i].set_title('North Pole', fontsize=fs+5, y=-0.12)
    ax[i].set_yticklabels(labels = ['180'])

    i = 1
    cs = ax[i].contourf(-Smx, Smx, SZq, levels=lvls, vmax=fhat.max())
    #ax[i].scatter(*Slamb, marker='x', color='r')
    ax[i].quiver(-Slamb[0], Slamb[1], -Sarrlamb[0], Sarrlamb[1], color='r', scale=Qq.scale, scale_units='width');
    ax[i].tick_params( direction = 'out' )
    ax_r = ax[i].secondary_yaxis('right')
    ax_r.tick_params(axis='y', direction='in', labelsize=fs)
    ax_r.set_yticks(ticks = [0])
    ax_r.set_yticklabels(labels = ['180'])
    ax[i].set_title('South Pole', fontsize=fs+5, y=-0.12)

    clb = fig.colorbar(cs, cax=ax[2], shrink=0.25, fraction=0.9);
    clb.ax.tick_params(labelsize=fs)

    title = '_'.join(bname.split('_')[:2]) + ' ' + lname + ' : k = {:.2f}'.format(k)
    fig.suptitle(title, fontsize=fs+10)

    fig.tight_layout();

    if writefig:
        #filename = oil_dst + bname + '_' + lname + '_emi_kde_N'
        fig.savefig(filename + '.jpg', dpi=72, format='jpg', pil_kwargs={'optimize':True}, bbox_inches='tight')
        plt.close()

def lambert_plot_X(k, fs=18, lvls=12, writefig=False, filename='file'):

    fig, ax = plt.subplots(1, 3, figsize=(15,8), gridspec_kw={"width_ratios":[1,1, 0.05]})

    for i in [0,1]:
        ax[i].set_aspect('equal')
        ax[i].plot(np.sqrt(2)*np.cos(theta), np.sqrt(2)*np.sin(theta), c='silver', lw=2);
        ax[i].tick_params(labelsize=fs)
        ax[i].set_xticks(ticks = [0])
        ax[i].set_yticks(ticks = [0])
        ax[i].set_xticklabels(labels = ['S'])

        ax_t = ax[i].secondary_xaxis('top')
        ax_t.tick_params(axis='x', direction='inout', labelsize=fs)
        ax_t.set_xticks(ticks = [0])
        ax_t.set_xticklabels(labels = ['N'])

    i = 0
    cs = ax[i].contourf(Wmx,Wmx, WZq, levels=lvls, vmax=fhat.max())
    #ax[i].scatter(*Nlamb, marker='x', color='r')
    Qq = ax[i].quiver(*Wlamb, *Warrlamb, color='r', scale_units='width')#, scale=13)
    Qq._init()
    assert isinstance(Qq.scale, float)

    ax[i].set_title('Front', fontsize=fs+5, y=-0.12)
    ax[i].set_yticklabels(labels = ['W'])

    i = 1
    cs = ax[i].contourf(-Emx, Emx, EZq, levels=lvls, vmax=fhat.max())
    ax[i].quiver(-Elamb[0], Elamb[1], -Earrlamb[0], Earrlamb[1], color='r', scale=Qq.scale, scale_units='width');
    ax[i].tick_params( direction = 'out' )
    ax_r = ax[i].secondary_yaxis('right')
    ax_r.tick_params(axis='y', direction='in', labelsize=fs)
    ax_r.set_yticks(ticks = [0])
    ax_r.set_yticklabels(labels = ['W'])
    ax[i].set_title('Back', fontsize=fs+5, y=-0.12)

    clb = fig.colorbar(cs, cax=ax[2], shrink=0.25, fraction=0.9);
    clb.ax.tick_params(labelsize=fs)

    title = '_'.join(bname.split('_')[:2]) + ' ' + lname + ' : k = {:.2f}'.format(k)
    fig.suptitle(title, fontsize=fs+10)

    fig.tight_layout();

    if writefig:
        #filename = oil_dst + bname + '_' + lname + '_emi_kde_W'
        fig.savefig(filename + '.jpg', dpi=72, format='jpg', pil_kwargs={'optimize':True}, bbox_inches='tight')
        plt.close();
