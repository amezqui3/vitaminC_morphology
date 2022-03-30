import os
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy import linalg
from scipy import special
from scipy.integrate import romberg

import tifffile as tf

import unionfind as UF

def persistence_with_UF(img, showfig=True, writefig=False, bname='file', dst='./', dpi=100):
    hist0,bins = np.histogram(img,bins=2**(img.dtype.itemsize*8),range=(0,2**(img.dtype.itemsize*8)))
    pers = sorted(UF.persistence(hist0[1:]),reverse=True)
    if showfig:
        plt.figure(figsize=(15,5))
        plt.plot(np.log(hist0[1:]+1), lw=3)
        plt.title(bname, fontsize=20);

        if writefig:
            filename = dst + bname + '.jpg'
            plt.savefig(filename, dpi=dpi, format='jpg', pil_kwargs={'optimize':True}, bbox_inches='tight');
            plt.close()
    return pers

def clean_zeroes(img):
    dim = img.ndim
    orig_size = img.size

    cero = list(range(2*dim))

    for k in range(dim):
        ceros = np.all(img == 0, axis = (k, (k+1)%dim))

        for i in range(len(ceros)):
            if(~ceros[i]):
                break
        for j in range(len(ceros)-1, 0, -1):
            if(~ceros[j]):
                break
        cero[k] = i
        cero[k+dim] = j+1

    img = img[cero[1]:cero[4], cero[2]:cero[5], cero[0]:cero[3]]

    print(round(100-100*img.size/orig_size),'% reduction from input')

    return img, cero

def get_comp_boxes(img, cero, thr, cutoff = 1e-2):
    labels,num = ndimage.label(img, structure=ndimage.generate_binary_structure(img.ndim, 1))
    print(num,'components')
    hist,bins = np.histogram(labels, bins=num, range=(1,num+1))
    regions = ndimage.find_objects(labels)
    sz_hist = np.sum(hist)
    argsort_hist = np.argsort(hist)[::-1]
    boxes = np.zeros((np.sum(hist/sz_hist > cutoff)+1, 7), dtype=int)
    print(boxes.shape)
    idx = 0
    for j in range(len(regions)):
        i = argsort_hist[j]
        r = regions[i]
        if(hist[i]/sz_hist > 1e-2):
            boxes[idx, :6] = r[0].start, r[1].start, r[2].start, r[0].stop, r[1].stop, r[2].stop
            idx += 1
        boxes[-1, :6] = cero[1], cero[2], cero[0], cero[4], cero[5], cero[3]
    boxes[:,-1] = thr

    return boxes

########################################################################
########################################################################

def adjust_box_values(dfboxes):
    boxes = dfboxes.values.copy()
    for i in range(len(boxes)-1):
        boxes[i,0] += boxes[-1,0]
        boxes[i,3] += boxes[-1,0]

        boxes[i,1] += boxes[-1,1]
        boxes[i,4] += boxes[-1,1]

        boxes[i,2] += boxes[-1,2]
        boxes[i,5] += boxes[-1,2]

    return boxes

def determine_mask(citrus, thr, its=4):
    mask = citrus.copy()
    mask[mask < thr] = 0
    mask[mask > 0] = 1
    structure = ndimage.generate_binary_structure(mask.ndim-1,mask.ndim-1)

    for k in range(mask.shape[2]):
        foo = mask[:,:,k]
        foo = ndimage.binary_dilation(foo, structure=structure, iterations=its)
        foo = ndimage.binary_fill_holes(foo)
        foo = ndimage.binary_erosion(foo, structure=structure, iterations=its)
        mask[:,:,k] = foo

    for k in range(mask.shape[0]):
        foo = mask[k, :, :]
        foo = ndimage.binary_dilation(foo, structure=structure, iterations=its)
        foo = ndimage.binary_fill_holes(foo)
        foo = ndimage.binary_erosion(foo, structure=structure, iterations=its)
        mask[k,:,:] = foo

    for k in range(mask.shape[1]):
        foo = mask[:, k, :]
        foo = ndimage.binary_dilation(foo, structure=structure, iterations=its)
        foo = ndimage.binary_fill_holes(foo)
        foo = ndimage.binary_erosion(foo, structure=structure, iterations=its)
        mask[:, k, :] = foo

    return mask

def polish_mask(img):

    box = img.copy()
    labels,num = ndimage.label(img, structure=ndimage.generate_binary_structure(img.ndim, 1))
    print(num,'components to polish')

    if num != 1:
        hist,bins = np.histogram(labels, bins=num, range=(1,num+1))
        sz_hist = np.sum(hist)
        argsort_hist = np.argsort(hist)[::-1]

        i = argsort_hist[0]
        mask = labels == i+1
        box[~mask] = 0

    return box

def plot_comp_cleaning(citrus, mask, cleaned, k, thr, lab=0, writefig=False, bname='file', pic_dst='./'):
    cmap = 'inferno'
    fig, ax = plt.subplots(2,2,figsize=(12, 10))

    j = (0,0)
    conv = ax[j].imshow(citrus[:,:,k], cmap=cmap, vmin=0)
    ax[j].axis('off')
    ax[j].set_title('Raw', fontsize=20)
    cbar = plt.colorbar(conv, ax=ax[j])
    cbar.ax.tick_params(labelsize=15)

    comp = citrus[:,:,k].copy()
    comp[comp < thr] = 0
    j = (0,1)
    conv = ax[j].imshow(comp, cmap=cmap, vmin=0)
    ax[j].axis('off')
    ax[j].set_title('Template', fontsize=20)
    cbar = plt.colorbar(conv, ax=ax[j])
    cbar.ax.tick_params(labelsize=15)

    j = (1,0)
    conv = ax[j].imshow(mask[:,:,k], cmap=cmap, vmin=0)
    ax[j].axis('off')
    ax[j].set_title('Masked', fontsize=20)
    cbar = plt.colorbar(conv, ax=ax[j])
    cbar.ax.tick_params(labelsize=15)

    j = (1,1)
    conv = ax[j].imshow(cleaned[:,:,k], cmap=cmap, vmin=0)
    ax[j].axis('off')
    ax[j].set_title('Thresholded', fontsize=20)
    cbar = plt.colorbar(conv, ax=ax[j])
    cbar.ax.tick_params(labelsize=15)

    plt.suptitle(bname + ', Comp {}'.format(lab), fontsize=25);

    if writefig:
        filename = pic_dst + bname + '_L{}_{}.jpg'.format(lab,k)
        plt.savefig(filename, dpi=96, format='jpg', pil_kwargs={'optimize':True}, bbox_inches='tight');
        plt.close()

########################################################################
########################################################################


def get_individual_threshold(img, showfig=False):
    pers = persistence_with_UF(img, showfig)
    print(pers)

    if len(pers) >= 3:
        peaks = np.zeros(3, dtype=int)
        for i in range(len(peaks)):
            peaks[i] = pers[i][2]
        thr = int(np.average(peaks, weights=np.array([7,3,1])))
    if len(pers) == 2:
        thr = 0.1*(7*pers[0][2] + 3*pers[1][2])
    if len(pers) == 1:
        thr = pers[0][2]

    return thr


def preprocess_segmentation(img, boundary, thr1=128, sigma=3):
    sub = img.copy()
    if boundary == 'upper':
        sub[sub < thr1] = 0
    elif boundary == 'lower':
        sub[sub > thr1] = 0

    blur = ndimage.gaussian_filter(sub, sigma=sigma, mode='constant', truncate=3, cval=0)

    return sub, blur

def plot4x4panel(img_list, ss, vmax=None, writefig=False, dst='./', bname='file'):

    rnum = 2
    cnum = len(img_list)//rnum
    fig, ax = plt.subplots(rnum,cnum,figsize=(5*cnum,5*rnum))
    print(rnum, cnum)

    if vmax is None:
        for idx in range(len(img_list)):
            i = idx//cnum
            j = idx%cnum
            ax[i,j].imshow(img_list[idx][ss], cmap='inferno', origin='lower');
    else:
        for idx in range(len(img_list)):
            i = idx//cnum
            j = idx%cnum
            ax[i,j].imshow(img_list[idx], cmap='inferno', origin='lower', vmax = vmax[idx]);

    plt.tight_layout();

    if writefig:
        filename = pic_dst + bname + '_rind0.jpg'
        plt.savefig(filename, dpi=96, format='jpg', pil_kwargs={'optimize':True}, bbox_inches='tight');

def get_largest_element(comp):
    labels,num = ndimage.label(comp, structure=ndimage.generate_binary_structure(comp.ndim, 1))
    print(num,'components')
    hist,bins = np.histogram(labels, bins=num, range=(1,num+1))
    argsort_hist = np.argsort(hist)[::-1]
    print(np.sort(hist)[::-1][:20])

    j = 0
    i = argsort_hist[j]
    mask = labels==i+1
    box0 = comp.copy()
    box0[~mask] = 0
    box0[box0 > 0] = 1

    return box0

def fill_component(comp, x=True, y=True, z=True):
    rcomp = comp.copy()
    rcomp[rcomp > 0] = 1

    if x:
        for k in range(rcomp.shape[0]):
            rcomp[k,:,:] = ndimage.binary_fill_holes(rcomp[k,:,:])
        print('Closed X')
    if y:
        for k in range(rcomp.shape[1]):
            rcomp[:,k,:] = ndimage.binary_fill_holes(rcomp[:,k,:])
        print('Closed Y')
    if z:
        for k in range(rcomp.shape[2]):
            rcomp[:,:,k] = ndimage.binary_fill_holes(rcomp[:,:,k])
        print('Closed Z')

    return rcomp

def fill_2d_line(img, a,b,c,d, k, disp=0):
    m = (b-d)/(a-c)
    B = b - a*m
    bar = disp + b
    for j in range(a,c):
        foo = disp + int(m*j + B)
        img[bar:foo, j, k] = 1
    return img

########################################################################
########################################################################

def collapse_dimensions(img):
    snaps = []
    for i in range(img.ndim):
        snaps.append(np.sum(img, axis=i))
    return snaps

def plot_collapse_dimensions(snaps, bname='bname', tissue='tissue', display=False, writefig=False, dst='./'):
    fig, ax = plt.subplots(1,len(snaps),figsize=(6*len(snaps),6))
    for i in range(len(snaps)):
        ax[i].imshow(snaps[i], cmap='inferno', origin='lower');
    plt.suptitle(bname + ' ' + tissue + ' collapse', fontsize=20);
    plt.tight_layout()

    if writefig:
        filename = dst + bname + '_' + '_'.join(tissue.split(' ')) + '.jpg'
        plt.savefig(filename, dpi=96, format='jpg', pil_kwargs={'optimize':True}, bbox_inches='tight');
        if not display:
            plt.close();

def tiff2coords(img, center=False):
    coords = np.nonzero(img)
    coords = np.vstack(coords).T
    if center:
        origin = -1*np.mean(coords, axis=0)
        coords = np.add(coords, origin)

    return coords

def plot_3Dprojections(seed, title='title', markersize=2, alpha=1, mk = '.', writefig=False, dst='./', dpi=125):
    axes = ['X','Y','Z']
    fig, ax = plt.subplots(1,3,figsize=(12,4))

    for i in range(3):
        proj = []
        for j in range(3):
            if j != i:
                proj.append(j)
        ax[i].scatter(seed[:,proj[1]], seed[:,proj[0]], s=markersize, color='y', alpha=alpha, marker=mk)
        ax[i].set_ylabel(axes[proj[0]])
        ax[i].set_xlabel(axes[proj[1]])
        ax[i].set_title(axes[i] + ' Projection')
        ax[i].set_aspect('equal', 'datalim');

    fig.suptitle(title, y=0.95, fontsize=20)
    plt.tight_layout();

    if writefig:
        filename = '_'.join(title.split(' '))
        plt.savefig(dst + filename + '.png', dpi=dpi, format='png', bbox_inches='tight',
                    facecolor='white', transparent=False)
        plt.close();

########################################################################
########################################################################

def label_and_rearrange(obinglands):
    numgeq1 = True

    struc = ndimage.generate_binary_structure(obinglands.ndim, 1)

    labels,num = ndimage.label(obinglands, structure=struc)
    print(num,'components')
    if num > 1:
        hist,bins = np.histogram(labels, bins=num, range=(1,num+1))
        argsort_hist = np.argsort(hist)[::-1]
        print(np.sort(hist)[::-1][:20])

        npz = np.hstack(([0],1+np.argsort(argsort_hist)))
        with np.nditer(labels, flags=['external_loop'], op_flags=['readwrite']) as it:
            for x in it:
                x[...] = npz[x]
    else:
        numgeq1 = False
        hist = np.arange(1)

    return labels, hist, numgeq1

def refine_oil_gland(binglands, maxsize, minsize, iters=1):
    struc = ndimage.generate_binary_structure(binglands.ndim, 1)
    obinglands = ndimage.binary_opening(binglands, structure=struc, iterations=iters)

    labels, hist, numgeq1 = label_and_rearrange(obinglands)
    if numgeq1:
        num = len(hist)

        above_out = np.sum(hist > maxsize)
        print('above_out',above_out)
        if above_out == 0:
            numgeq = False

        below_out = np.sum(hist < minsize)
        print('below_out', below_out)

        normalsized = np.arange(above_out, num-below_out)
        largesized = np.arange(0, above_out)
        smallsized = np.arange(num - below_out+1, num+1)
        print(len(normalsized), len(largesized))

    else:
        normalsized = np.array([1])
        largesized = np.arange(0)

    return obinglands, labels, normalsized, largesized, numgeq1

# Taken from https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python/68385697#68385697
def buf_count_newlines_gen(fname):
    def _make_gen(reader):
        while True:
            b = reader(2 ** 16)
            if not b: break
            yield b

    with open(fname, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
    return count

########################################################################
########################################################################

def ellipsoid(lamb, phi, ax=3, ay=2, az=1):

    ex2 = (ax*ax - az*az)/(ax*ax)
    ey2 = (ay*ay - az*az)/(ay*ay)
    ee2 = (ax*ax - ay*ay)/(ax*ax)

    nu = ax/(np.sqrt(1 - ex2*np.sin(phi)**2 - ee2*(np.cos(phi)*np.sin(lamb))**2))

    return np.array([nu*np.cos(phi)*np.cos(lamb),
                     nu*(1-ee2)*np.cos(phi)*np.sin(lamb),
                     nu*(1-ex2)*np.sin(phi)])

def spine_based_alignment(spine, title='bname_L99', savefig=False, dst='./'):
    scoords = tiff2coords(spine)
    size = 7
    skip = np.array([size,size,size])
    downsample = scoords[np.all(np.fmod(scoords, skip) == 0, axis=1), :]
    downsample = downsample - np.mean(scoords, axis = 0)

    _, _, vh = np.linalg.svd(downsample, full_matrices=False)
    salign = np.matmul(downsample, np.transpose(vh))

    plot_3Dprojections(salign, title=title, writefig=savefig, dst=dst, dpi=96)

    return vh

def conic_fitting(datapoints):
    x = datapoints[:,0]
    y = datapoints[:,1]
    z = datapoints[:,2]

    J = np.column_stack((x**2, y**2, z**2, x*y, x*z, y*z, x, y, z))
    K = np.ones_like(x)

    vec = (np.linalg.inv((J.T @ J))@J.T)@K
    vec = np.append(vec, -1)
    Amat=np.array(
       [
       [ vec[0],     vec[3]/2.0, vec[4]/2.0, vec[6]/2.0 ],
       [ vec[3]/2.0, vec[1],     vec[5]/2.0, vec[7]/2.0 ],
       [ vec[4]/2.0, vec[5]/2.0, vec[2],     vec[8]/2.0 ],
       [ vec[6]/2.0, vec[7]/2.0, vec[8]/2.0, vec[9]     ]
       ])

    return Amat

def params_from_mat(A):
    A3 = A[:3,:3]
    offset = A[3,:3]

    center = -np.linalg.inv(A3)@offset
    print('Center at')
    print(center)
    Tofs = np.identity(4)
    Tofs[3, :3] = center

    R = Tofs @ (A @ Tofs.T)
    R3 = R[:3,:3]
    s1 = - R[3,3]
    R3S = R3/s1
    (eigvals,eigvecs) = np.linalg.eig(R3S)

    recip = 1.0/np.abs(eigvals)
    axes=np.sqrt(recip)
    print('Axes gains')
    print(axes)
    print('Rotation')
    print(eigvecs)

    return center, axes, eigvecs

def cartesian_to_polar(coords):
    s_xx_yy = np.sqrt(coords[:,1]**2 + coords[:,2]**2)
    norms = np.sqrt(np.sum(coords**2, axis=1))

    latitud = np.zeros_like(norms)
    longitude = np.zeros_like(norms)
    polar = np.zeros_like(norms)
    azimuth = np.zeros_like(norms)

    poles = s_xx_yy < 1e-10

    latitud = np.arcsin(coords[:,0]/norms)
    longitude[~poles] = np.arccos(coords[~poles,2]/s_xx_yy[~poles])*(np.sign(coords[~poles,1]))

    polar = np.arccos(coords[:,0]/norms)
    azimuth[~poles] = np.arccos(coords[~poles,2]/s_xx_yy) + (np.sign(coords[~poles,1]) > 0).astype(int)*np.pi

    return [norms, latitud, longitude, azimuth, polar]

def sample_kde(N, kde):
    rng = np.random.RandomState(42)
    sample = kde.sample(N, rng)
    sample[sample[:,0] > 2*np.pi,0] = 4*np.pi - sample[sample[:,0] > 2*np.pi,0]
    sample[:,0] = np.abs(sample[:,0])
    sample[sample[:,1] > 0.5*np.pi,1] = np.pi - sample[sample[:,1] > 0.5*np.pi,1]
    sample[sample[:,1] < -.5*np.pi,1] = -np.pi-sample[sample[:,1] < -.5*np.pi,1]

    return sample

def cyl_overlap(longitude, latitude, overlap=np.pi/8):
    heatmap_lon = np.hstack((longitude-2*np.pi, longitude, longitude+2*np.pi))
    heatmap_lat = np.tile(latitude, 3)
    heatmask = (heatmap_lon < np.pi + overlap) & (heatmap_lon > -np.pi - overlap)
    heatmap = np.column_stack((heatmap_lon[heatmask], heatmap_lat[heatmask]))

    return heatmap, heatmask

########################################################################
########################################################################

def ellipsoidIntArea(A, B, C, start=0, verbose=False):

    a,b,c = A,B,C

    delta = 1 - c*c/(a*a)
    epsilon = 1 - c*c/(b*b)

    def int_fun(s):
        m = epsilon*(1 - s*s)/(1-delta*s*s)
        return np.sqrt(1 - delta*s*s)*special.ellipe(m)

    results = romberg(int_fun, start/a, 1, vec_func=True, show=verbose)

    area = 4*a*b*results
    if start == 0:
        area *= 2
    if verbose:
        print("{:.2f} {:.4f}".format(start, round(area, 4)))

    return area

def ellipsoidTotArea(a,b,c, verbose=False):

    #if (a == b) and (b == c):
    #    return 4*np.pi*a*a
    #c, b, a = np.sort([A,B,C])

    t = np.arccos(c/a)
    s = np.arccos(c/b)
    k = np.sin(s)/np.sin(t)

    area = c*c/(a*a)*special.ellipkinc(t, k*k) + np.sin(t)*np.sin(t)*special.ellipeinc(t, k*k)
    area *= 2*np.pi*a*b/np.sin(t)
    area += 2*np.pi*c*c

    if verbose:
        print("{:.4f}".format(round(area, 4)))

    return area

def vonmises_kde(data, kappa, n_bins=100):
    bins = np.linspace(-np.pi, np.pi, n_bins)
    x = np.linspace(-np.pi, np.pi, n_bins)
    # integrate vonmises kernels
    kde = np.exp(kappa*np.cos(x[:, None]-data[None, :])).sum(1)/(2*np.pi*special.i0(kappa))
    kde /= np.trapz(kde, x=bins)
    return bins, kde

########################################################################
#
# Adapted from
#    Necula, Ioana; Diaz-Toca, Gema Maria; Marin, Leandro (2020),
#    "Code C++ - transforming cartesian coordinates into geodetic
#     coordinates", Mendeley Data, V2, doi: 10.17632/s5f6sww86x.2
#
# Made it numpy friendly, and now it can take either scalars or
# arrays as inputs
#
########################################################################

#****************************************************************
# ALGORITHM : numerical solution to polynomial equation         *
# INPUT : Polynomial B = B[6] x^6 + B[5] x^5 + ... + B[0]       *
#         Initial point x0                                      *
#         Maximum error                                         *
# OUTPUT: root found after applying Newton-Raphson method to B  *
#         The function returns the value when the correction    *
#         is smaller than error.                                *
#****************************************************************/

def polSolve(B, x0, error):
    iters = 0
    corr = 1e10;
    while((np.max(np.abs(corr)) > error) & (iters < 50)):
        f  = B[6];
        fp = B[6]
        f  = x0*f  + B[5];
        fp = x0*fp + f;
        f  = x0*f  + B[4];
        fp = x0*fp + f;
        f  = x0*f  + B[3];
        fp = x0*fp + f;
        f  = x0*f  + B[2];
        fp = x0*fp + f;
        f  = x0*f  + B[1];
        fp = x0*fp + f;
        f  = x0*f  + B[0];
        corr = f/fp;
        x0-= corr;
        iters += 1
    #print('NR Iterations required:', iters)
    return x0;

#****************************************************************
# ALGORITHM : Horner's method to compute B(x-c) in terms of B(x)*
# INPUT : Polynomial B = B[6] x^6 + B[5] x^5 + ... + B[0]       *
# OUTPUT: Polynomial BB such that B(x-c) = BB(x)                *
#****************************************************************/

def horner(B, c):
    #// B(x-c) = BB(x)
    BB = np.copy(B)
    for i in range(len(BB)):
        for j in range(len(BB)-2, i-1, -1):
              BB[j] -= BB[j+1]*c;
    return BB

# The algorithm above assumes xG, yG, zG >= 0, in the first octant
# Reflect footpoints accordingly

def correct_signs(geodetic, signs):

    geodetic[0] = np.atleast_1d(geodetic[0])

    where = np.atleast_1d(signs[0] & signs[1])
    geodetic[0][where] -= np.pi

    where = np.atleast_1d(signs[0] & ~signs[1])
    geodetic[0][where] = -geodetic[0][where] + np.pi

    where = np.atleast_1d(~signs[0] & signs[1])
    geodetic[0][where] *= -1

    geodetic[1] = np.where(signs[2], -geodetic[1], geodetic[1])

    return np.asarray(geodetic)

#******************************************************************
# ALGORITHM : Cartesian into Geodetic I                           *
# INPUT:                                                          *
#       semiaxes of the celestial body: ax>ay>az                  *
#       cartesian coordinates of the considered point in the      *
#           first octant: xG, yG, zG with (xG,yG,zG)<>(0,0,0)     *
#       error. Values smaller than error treated as 0.0           *
# OUPUT:                                                          *
#       values latitude, longitude and altitude with the geodetic *
#                           coordinates of the considered point   *
# CALLING the procedure CartesianIntoGeodeticI:                   *
#       CartesianIntoGeodeticI(ax,ay,az,xG,yG,zG);                *
#******************************************************************/

def CartesianIntoGeodeticI(ax, ay, az, xG, yG, zG, error=1e-8):
    xG = np.atleast_1d(np.abs(xG))
    yG = np.atleast_1d(np.abs(yG))
    zG = np.atleast_1d(np.abs(zG))

    # /* Computations independent of xG,yG,zG. They can be precomputed, if necessary. */

    ax2 = ax*ax;
    ay2 = ay*ay;
    az2 = az*az;
    ax4 = ax2*ax2;
    ay4 = ay2*ay2;
    az4 = az2*az2;
    b5 = 2*(ax2+ay2+az2);
    b4 = ax4 + 4*ax2*ay2 + ay4 + 4*ax2*az2 + 4*ay2*az2 + az4;
    b3 = 2*ax4*ay2 + 2*ax2*ay4 + 2*ax4*az2 + 8*ax2*ay2*az2 + 2*ay4*az2 + 2*ax2*az4 + 2*ay2*az4;
    b3x = - 2*ax2*ay2 - 2*ax2*az2;
    b3y = - 2*ax2*ay2 - 2*ay2*az2;
    b3z = - 2*ay2*az2 - 2*ax2*az2;
    b2 = 4*ax4*ay2*az2 + 4*ax2*ay4*az2 + ax4*az4 + 4*ax2*ay2*az4 + ax4*ay4 + ay4*az4;
    b2x = -ax2*ay4 -4*ax2*ay2*az2 -ax2*az4;
    b2y = -ax4*ay2 -4*ax2*ay2*az2 -ay2*az4;
    b2z = -ax4*az2 -4*ax2*ay2*az2 -ay4*az2;
    b1 = 2*ax4*ay4*az2 + 2*ax4*ay2*az4 + 2*ax2*ay4*az4;
    b1x = - 2*ax2*ay4*az2 - 2*ax2*ay2*az4;
    b1y = - 2*ax4*ay2*az2 - 2*ax2*ay2*az4;
    b1z = - 2*ax4*ay2*az2 - 2*ax2*ay4*az2;
    b0 = ax4*ay4*az4;
    b0x = - ax2*ay4*az4;
    b0y = - ax4*ay2*az4;
    b0z = - ax4*ay4*az2;
    eec=(ax2-ay2)/ax2;
    exc=(ax2-az2)/ax2;

    #/* Computations dependant of xG, yG, zG */
    xg2 = xG*xG;
    yg2 = yG*yG;
    zg2 = zG*zG;
    aux = xg2/ax2 + yg2/ay2 + zg2/az2;
    guess = (xg2+yg2+zg2)/3.0

    B = np.zeros((7, len(xG)))

    B[6] = 1.0;
    B[5] = b5;
    B[4] = b4 - (ax2*xg2+ay2*yg2+az2*zg2);
    B[3] = b3+b3x*xg2+b3y*yg2+b3z*zg2;
    B[2] = b2+b2x*xg2+b2y*yg2+b2z*zg2;
    B[1] = b1+b1x*xg2+b1y*yg2+b1z*zg2;
    B[0] = b0+b0x*xg2+b0y*yg2+b0z*zg2;

    B = np.atleast_2d(B)

    xE = np.copy(xG)
    yE = np.copy(yG)
    zE = np.copy(zG)

    # // The point is outside the ellipsoid
    outE = aux > (1.0 + error)
    if np.sum(outE) > 0:
        k = polSolve(B[:,outE],guess[outE],error);
        xE[outE] = ax2*xG[outE]/(ax2+k)
        yE[outE] = ay2*yG[outE]/(ay2+k)
        zE[outE] = az2*zG[outE]/(az2+k)

    # // The point  is inside the ellipsoid and zG>0
    inE = aux < (1.0 - error)
    where = inE & (zG > error)
    if np.sum(where) > 0:
        BB = np.atleast_2d(horner(B[:,where], az2 )) #// B(x-az2) = BB(x)
        k = polSolve(BB, guess[where] + az2, error) - az2;
        xE[where] = ax2*xG[where]/(ax2+k)
        yE[where] = ay2*yG[where]/(ay2+k)
        zE[where] = az2*zG[where]/(az2+k)

    # The point  is inside the ellipsoid and zG=0, yG > 0, xG > 0
    where = inE & (zG < error) & (xG > error) & (yG > error)
    if np.sum(where) > 0:
        BB = np.atleast_2d(horner(B[:,where],ay2))
        k = polSolve(BB, guess[where]+ay2, error) - ay2
        xE[where] = (ax2*xG[where]/(ax2+k))
        yE[where] = (ay2*yG[where]/(ay2+k))
        zE[where] = 0.0;

    where = inE & (zG < error) & (xG < error) & (yG > error)
    if np.sum(where) > 0:
        xE[where] = 0.0;
        yE[where] = ay;
        zE[where] = 0.0;

    where = inE & (zG < error) & (xG > error) & (yG < error)
    if np.sum(where) > 0:
        xE[where] = ax
        yE[where] = 0.0
        zE[where] = 0.0

    #// Computing longitude

    longitude = np.where(xE > error, np.arctan(yE/((1.0-eec)*xE)), np.pi/2.0)

    #// Computing latitude

    norm = np.linalg.norm(np.vstack((xE*(1.0-eec),yE)), axis=0)
    latitude = np.where( (xE > error) | (yE > error), np.arctan((1.0-eec)/(1.0-exc)*zE/norm), np.pi/2.)

    #// Computing altitude

    normalV = np.vstack((xE-xG,yE-yG,zE-zG))
    altitude = np.linalg.norm(normalV, axis=0)
    altitude = np.where(aux >= 1, altitude, -altitude)

    geodetic = np.vstack((longitude, latitude, altitude))
    footpoints = np.vstack((xE, yE, zE))

    return geodetic, footpoints

########################################################################
########################################################################

def ell_algebraic_fit(datapoints):
    X, Y, Z = datapoints
    D = np.column_stack((X**2, Y**2, Z**2, X*Y, X*Z, Y*Z, X, Y, Z))
    i1 = np.ones_like(X)
    c = (np.linalg.inv((D.T @ D))@D.T)@i1
    return c

def get_ell_params_from_vector(c, ax_guess=None):
    if ax_guess is None:
        ax_guess = [0,1,2]
    ax_guess = np.asarray(ax_guess)

    Cmat = np.array([
        [2*c[0], c[3]  , c[4]  ],
        [c[3],   2*c[1], c[5]  ],
        [c[4],   c[5]  , 2*c[2]]
    ])
    t = -np.linalg.inv(Cmat) @ c[6:9]

    D = 1 + np.sum(c[:3]*(t**2)) + c[3]*t[0]*t[1] + c[4]*t[0]*t[2] + c[5]*t[1]*t[2]

    Q = D*np.linalg.inv(.5*Cmat)
    eigvals, R = np.linalg.eigh(Q)
    axis = np.sqrt(eigvals)[ax_guess]
    for i,cc in enumerate(np.column_stack((np.argmax(np.abs(R), axis = 0), [0,1,2]))):
        if(R[tuple(cc)] < 0 ):
            R[:,i] = -R[:,i]
    R = R[:, ax_guess]
    R = R.T
    t = t

    thetax = np.arctan(-R[2,1]/R[2,2])
    thetay = np.arctan( R[2,0]/(np.sqrt(R[0,0]**2 + R[1,0]**2)))
    thetaz = np.arctan(-R[1,0]/R[0,0])

    ell_params = {'center': t,
                  'axes': axis,
                  'rotation': R,
                  'theta': np.array([thetax, thetay, thetaz])}
    return ell_params

def get_footpoints(datapoints, ell_params, footpoints='geodetic', error=1e-8):
    t = ell_params['center']
    R = ell_params['rotation']
    axes = ell_params['axes']

    UVW = R @ (datapoints - t.reshape(-1,1))
    signs = UVW < 0

    if footpoints == 'geodetic':
        geocoord, footpoints = CartesianIntoGeodeticI(*axes, *UVW, error=error)
    elif footpoints == 'geocentric':
        geocoord, footpoints = CartesianIntoGeocentric(*axes, *UVW, error=error)
    else:
        print('Only "geodetic" or "geocentric" footpoint values are valid!')
        return -1, -1

    geocoord = correct_signs(geocoord, signs)

    uvw = np.where(signs, -1, 1)*footpoints
    xyz = t.reshape(-1,1) + R.T @ uvw

    return geocoord, xyz

def geoid_heights(datapoints, xyz):
    Nis = np.sqrt(np.sum((datapoints - xyz)**2, axis=0))
    rms = np.sqrt(np.sum(Nis**2)/len(Nis))

    return np.min(Nis), np.mean(Nis), np.max(Nis), rms

def ell_rho(axes):
    a,b,c = 1./(np.asarray(axes)**2)
    I = a + b + c
    J = a*b + b*c + c*a
    #K = np.linalg.det(np.array([[a, h, g], [h, b, f], [g, f, c] ]))
    rho = (4*J - I*I)/(a**2 + b**2 + c**2)
    return rho

def ell_algebraic_fit2(datapoints, k=4):

    x,y,z = datapoints
    C1 = np.diag([-1, -1, -1, -k, -k, -k])
    C1[0,1:3] = .5*k - 1
    C1[1:3, 0] = .5*k - 1
    C1[1,2] = .5*k - 1
    C1[2,1] = .5*k - 1
    D = np.vstack((x**2, y**2, z**2, 2*x*y, 2*x*z, 2*y*z, 2*x, 2*y, 2*z, np.ones_like(x)))

    DDT = D@D.T
    S11 = DDT[:6, :6]
    S12 = DDT[:6, 6:]
    S22 = DDT[6:, 6:]

    M = (np.linalg.inv(C1))@(S11 - S12@np.linalg.inv(S22)@(S12.T))
    eigvals, eigvecs = np.linalg.eig(M)
    v1 = eigvecs[:, np.argmax(eigvals)]
    v2 = -np.linalg.inv(S22)@(S12.T)@v1

    v = -np.hstack((v1,v2))/v2[-1]

    a,b,c,f,g,h, p,q,r, d = v
    I = a + b + c
    J = a*b + b*c + c*a - f*f/4 - g*g/4 - h*h/4
    aux = k*J - I*I

    return v, aux > 0

def CartesianIntoGeocentric(ax, ay, az, xG, yG, zG, error=1e-8):
    xG = np.atleast_1d(np.abs(xG))
    yG = np.atleast_1d(np.abs(yG))
    zG = np.atleast_1d(np.abs(zG))

    ax2 = ax*ax;
    ay2 = ay*ay;
    az2 = az*az;
    eec=(ax2-ay2)/ax2;
    exc=(ax2-az2)/ax2;

    #/* Computations dependant of xG, yG, zG */
    xg2 = xG*xG;
    yg2 = yG*yG;
    zg2 = zG*zG;
    aux = np.sqrt(xg2/ax2 + yg2/ay2 + zg2/az2)
    aux = np.where(aux > error, aux, 1)

    xE = np.copy(xG)/aux
    yE = np.copy(yG)/aux
    zE = np.copy(zG)/aux

    #// Computing longitude

    longitude = np.where(xE > error, np.arctan(yE/((1.0-eec)*xE)), np.pi/2.0)

    #// Computing latitude

    norm = np.linalg.norm(np.vstack((xE*(1.0-eec),yE)), axis=0)
    latitude = np.where( (xE > error) | (yE > error), np.arctan((1.0-eec)/(1.0-exc)*zE/norm), np.pi/2.)

    #// Computing altitude

    normalV = np.vstack((xE-xG,yE-yG,zE-zG))
    altitude = np.linalg.norm(normalV, axis=0)
    altitude = np.where(aux >= 1, altitude, -altitude)

    geodetic = np.vstack((longitude, latitude, altitude))
    footpoints = np.vstack((xE, yE, zE))

    return geodetic, footpoints

########################################################################
########################################################################

def plot_ell_comparison(cglands, eglands, ecoords, title='title', sidestep=1, fs=20, alpha = 0.5, markersize=2, savefig=False, filename='file'):
    axl = ['X','Y','Z']
    fig, ax = plt.subplots(1,3,figsize=(15,5))

    for i in range(3):
        proj = []
        for j in range(3):
            if j != i:
                proj.append(j)

        ax[i].plot(cglands[proj[0]] - sidestep, cglands[proj[1]], '.', ms=markersize, c='cyan', alpha=alpha)
        ax[i].plot(eglands[proj[0]] + sidestep, eglands[proj[1]], '.', ms=markersize, c='orange', alpha=alpha)
        ax[i].plot(ecoords[proj[0]] + sidestep, ecoords[proj[1]], '.', ms=2*markersize, c='gray', alpha=.75*alpha)

        ax[i].set_ylabel(axl[proj[1]], fontsize=fs)
        ax[i].set_xlabel(axl[proj[0]], fontsize=fs)
        ax[i].set_title(axl[i] + ' Projection')
        ax[i].set_aspect('equal', 'datalim');

    fig.suptitle(title, y=0.95, fontsize=fs)
    fig.tight_layout();
    if savefig:
        fig.savefig(filename + '.jpg', format='jpg', bbox_inches = 'tight', pil_kwargs={'optimize': True}, dpi=100)
        plt.close();

def plot_lambert_cylindrical(geodetic, latitude, title, alpha=.2, marker='o', ms=50, fs=20, savefig=False, filename='file'):
    skin_map, _ = cyl_overlap(*geodetic[:2])

    plt.figure(figsize=(18,8))
    plt.scatter(skin_map[:,0], np.sin(skin_map[:,1]), alpha=alpha, marker=marker, s=ms, color='orange')
    plt.xlabel('Longitude', fontsize=fs); plt.ylabel('Latitude', fontsize=fs);
    plt.title(title, fontsize=fs);
    plt.ylim((-1,1)); plt.axvline(x=-np.pi, c='darkgray', ls=':');plt.axvline(x=np.pi, c='darkgray', ls=':')
    for y in np.sin(latitude):
        plt.axhline(y=y, c='silver', ls='--')
    plt.axhline(y=0, c='blue')
    plt.margins(y=0);

    if savefig:
        plt.savefig(filename + '.jpg', format='jpg', bbox_inches = 'tight', pil_kwargs={'optimize': True}, dpi=100);
        plt.close();

def plot_lambert_azimuthal(geodetic, latitude, title, alpha=.2, marker='o', ms=50, fs=20, savefig=False, filename='file'):
    NlambertX = 2*np.sin(.25*np.pi - .5*geodetic[1])*np.cos(geodetic[0])
    NlambertY = 2*np.sin(.25*np.pi - .5*geodetic[1])*np.sin(geodetic[0])
    SlambertX = 2*np.sin(.25*np.pi + .5*geodetic[1])*np.cos(geodetic[0])
    SlambertY = 2*np.sin(.25*np.pi + .5*geodetic[1])*np.sin(geodetic[0])

    fig, ax = plt.subplots(1,2,figsize=(12,7), sharex=True, sharey=True)

    foo = np.linspace(-np.pi, np.pi, 200)
    ax[0].scatter(NlambertX, NlambertY, alpha=alpha, marker=marker, s=ms, color='orange')
    ax[1].scatter(SlambertX, SlambertY, alpha=alpha, marker=marker, s=ms, color='orange')
    for i in [0,1]:
        for radius in 2*np.sin(.25*np.pi - .5*latitude):
            ax[i].plot(radius*np.cos(foo), radius*np.sin(foo), c='silver', ls='--')
        radius = np.sqrt(2)
        ax[i].plot(radius*np.cos(foo), radius*np.sin(foo), c='blue')
        ax[i].set_ylim((-2,2)); ax[i].set_xlim((-2,2))
        ax[i].set_aspect('equal')
    ax[0].set_xlabel('North pole', fontsize=fs);
    ax[1].set_xlabel('South pole', fontsize=fs);
    fig.suptitle(title, fontsize=fs, y=0.9);
    fig.tight_layout();

    if savefig:
        fig.savefig(filename + '.jpg', format='jpg', bbox_inches = 'tight', pil_kwargs={'optimize': True}, dpi=100);
        plt.close();
