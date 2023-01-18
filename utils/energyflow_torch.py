import torch
import numpy as np
import six
import math


## ASSUMING PHYSICS CONVENTION of Jets Tensor: Shape: (Jets, Particles, Features)


## ENERGY MOVER DISTANCE HELPER FUCTIONS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_pi = torch.Tensor([math.pi]).to(device)
torch_two_pi = torch_pi * 2

def process_for_emd(x, pt_scaling=1., periodic_phi=False):
    pts, coords = x[...,0], x[...,1:]
    pts = pts * pt_scaling
    # handle phi periodicity 
    if periodic_phi:
        y_col = 0
        phi_col = 1
        coords_clone = coords.clone()
        coords_clone[...,y_col] = coords[...,y_col]
        coords_clone[...,phi_col] = torch.fmod(coords[...,phi_col], torch_two_pi)
        return pts, coords_clone   
    return pts, coords

def cdist_emd(X, Y, periodic_phi=False):
    R = 1.      # scaling R from emd function
    y_col = 0
    phi_col = 1
    # assuming periodic phi
    if periodic_phi:
        d_phis = torch_pi - torch.abs(torch_pi - torch.abs(X[...,phi_col].unsqueeze(2) - Y[...,phi_col].unsqueeze(1)))
    else:
        d_phis = X[...,phi_col].unsqueeze(2) - Y[...,phi_col].unsqueeze(1)
    d_ys = X[...,y_col].unsqueeze(2) - Y[...,y_col].unsqueeze(1)
    out = torch.sqrt(torch.clamp_min(d_ys**2 + d_phis**2, 1e-8))
    out = out / R
    
    return out




# JET FUNCTIONS


def jet_masses(jets_tensor): # in format (jets, particles, features)
    jets_p4s = torch_p4s_from_ptyphi(jets_tensor)
    masses = torch_ms_from_p4s(jets_p4s.sum(axis=1))
    return masses

def jet_pts(jets_tensor): # in format (jets, particles, features)
    jets_p4s = torch_p4s_from_ptyphi(jets_tensor)
    pts = torch_pts_from_p4s(jets_p4s.sum(axis=1))
    return pts

def jet_ys(jets_tensor):
    jets_p4s = torch_p4s_from_ptyphi(jets_tensor)
    ys = torch_ys_from_p4s(jets_p4s.sum(axis=1))
    return ys

def jet_etas(jets_tensor):
    jets_p4s = torch_p4s_from_ptyphi(jets_tensor)
    etas = torch_etas_from_p4s(jets_p4s.sum(axis=1))
    return etas

def jet_phis(jets_tensor):
    jets_p4s = torch_p4s_from_ptyphi(jets_tensor)
    phis = torch_phis_from_p4s(jets_p4s.sum(axis=1), phi_ref=0)
    return phis

def jet_mults(jets_tensor):   # how many particles per jet - does not consider zero padding! torch.count_zero in release 1.8
    mults = torch.ones(jets_tensor.size()[0]) * jets_tensor.size()[1]
    return mults


# PARTICLE FUNCTIONS


def torch_p4s_from_ptyphi_perm(ptyphi):
    # get pts, ys, phis
    ptyphi = ptyphi.permute(0,2,1)
    #ptyphi = torch.Tensor(ptyphi).float()
    pts, ys, phis = (ptyphi[...,0,np.newaxis], 
                     ptyphi[...,1,np.newaxis], 
                     ptyphi[...,2,np.newaxis])

    Ets = torch.sqrt(pts**2) #  + ms**2) # everything assumed massless
    p4s = torch.cat((Ets*torch.cosh(ys), pts*torch.cos(phis), 
                          pts*torch.sin(phis), Ets*torch.sinh(ys)), axis=-1)
    return p4s.permute(0,2,1)


# p4s
def torch_p4s_from_ptyphi(ptyphi):
    # get pts, ys, phis
    #ptyphi = torch.Tensor(ptyphi).float()
    pts, ys, phis = (ptyphi[...,0,np.newaxis], 
                     ptyphi[...,1,np.newaxis], 
                     ptyphi[...,2,np.newaxis])

    Ets = torch.sqrt(pts**2) #  + ms**2) # everything assumed massless
    p4s = torch.cat((Ets*torch.cosh(ys), pts*torch.cos(phis), 
                          pts*torch.sin(phis), Ets*torch.sinh(ys)), axis=-1)
    return p4s


def torch_ptyphi_from_p4s(p4s, phi_ref=None, mass=False):   # assmuse structure (batch, points, feats)
    if p4s.shape[-1] != 4:
        raise ValueError("Last dimension of 'p4s' must have size 4.")
    out = torch.zeros(p4s.shape[:-1] + (4 if mass else 3,), dtype=float).to(p4s.device)
    out[...,0] = torch_pts_from_p4s(p4s)
    out[...,1] = torch_ys_from_p4s(p4s)
    out[...,2] = torch_phis_from_p4s(p4s, phi_ref, _pts=out[...,0])
    #if mass:
        #out[...,3] = ms_from_p4s(p4s)
    return out




# p3s
def torch_p3s_from_ptyphi(ptyphi):  # assumes structure (batch, points, feats)
    # get pts, ys, phis
    #ptyphi = torch.Tensor(ptyphi).float()
    pts, ys, phis = (ptyphi[...,0,np.newaxis], 
                     ptyphi[...,1,np.newaxis], 
                     ptyphi[...,2,np.newaxis])

    Ets = torch.sqrt(pts**2) #  + ms**2) # everything assumed massless
    p3s = torch.cat((pts*torch.cos(phis), 
                          pts*torch.sin(phis), Ets*torch.sinh(ys)), axis=-1)
    return p3s


def torch_ptyphi_from_p3s(p3s, device="cuda", phi_ref=None, mass=False, phi_add2pi=True):   # assmuse structure (batch, points, feats=px,py,pz)
    # calc energy = sqrt(p_x**2 + py**2 + p_z**2)   (massless case)
    if not mass:
        E = torch.sqrt(p3s[...,0]**2 + p3s[...,1]**2 + p3s[...,2]**2)[...,np.newaxis]
    else:
        raise ValueError("here only defined for massless case")
    
    p4s = torch.cat((E, p3s), axis=-1)
    if p4s.shape[-1] != 4:
        raise ValueError("Last dimension of 'p4s' must have size 4.")
    out = torch.zeros(p4s.shape[:-1] + (4 if mass else 3,), dtype=float).to(device)
    out[...,0] = torch_pts_from_p4s(p4s)
    out[...,1] = torch_ys_from_p4s(p4s)
    out[...,2] = torch_phis_from_p4s(p4s, phi_ref, _pts=out[...,0], phi_add2pi=phi_add2pi)
    return out.float()






def torch_pt2s_from_p4s(p4s):
    return p4s[...,1]**2 + p4s[...,2]**2


def torch_pts_from_p4s(p4s):
    return torch.sqrt(torch_pt2s_from_p4s(p4s))


def torch_m2s_from_p4s(p4s):
    return p4s[...,0]**2 - p4s[...,1]**2 - p4s[...,2]**2 - p4s[...,3]**2


def torch_ms_from_p4s(p4s):
    m2s = torch_m2s_from_p4s(p4s)
    return torch.sign(m2s)*torch.sqrt(torch.abs(m2s))


def torch_ys_from_p4s(p4s):
    ## RAPIDITY
    out = torch.zeros(p4s.shape[:-1], device=p4s.device).float()
    p4s = p4s.float()
    
    nz_mask = torch.any(p4s != 0., axis=-1)
    nz_p4s = p4s[nz_mask]
    ratio = nz_p4s[...,3]/(nz_p4s[...,0])
    ratio = torch.clamp(ratio, -0.99999, 0.99999) ## to avoid nans from atanh --> not sure if good practice..
    out[nz_mask] = torch.atanh(ratio)
    #out = torch.atanh(p4s[...,3]/(p4s[...,0]+eps))
    return out


def torch_etas_from_p4s(p4s):
    ## PSEUDO-RAPIDITY
    out = torch.zeros(p4s.shape[:-1], device=device).float()
    nz_mask = torch.any(p4s != 0., axis=-1)
    nz_p4s = p4s[nz_mask]
    out[nz_mask] = torch.atanh(nz_p4s[...,3]/torch.sqrt(nz_p4s[...,1]**2 + nz_p4s[...,2]**2 + nz_p4s[...,3]**2))

    return out


def torch_phi_fix(phis, phi_ref, copy=False):
    TWOPI = 2*np.pi

    diff = (phis - phi_ref)

    new_phis = torch.copy(phis) if copy else phis
    new_phis[diff > np.pi] -= TWOPI
    new_phis[diff < -np.pi] += TWOPI

    return new_phis


def torch_phis_from_p4s(p4s, phi_ref=None, _pts=None, phi_add2pi=True):
    # get phis
    phis = torch.atan2(p4s[...,2], p4s[...,1])
    if phi_add2pi:
        phis[phis<0] += 2*np.pi

    # ensure close to reference value
    if phi_ref is not None:
        if isinstance(phi_ref, six.string_types) and phi_ref == 'hardest':
            ndim = phis.ndim

            # here the particle is already phi fixed with respect to itself
            if ndim == 0:
                return phis

            # get pts if needed (pt2s are fine for determining hardest)
            hardest = torch.argmax(_pts, axis=-1)

            # indexing into vector
            if ndim == 1:
                phi_ref = phis[hardest]

            # advanced indexing
            elif ndim == 2:
                phi_ref = phis[torch.arange(len(phis)), hardest]

            else:
                raise ValueError("'p4s' should not have more than three dimensions.")

        phis = torch_phi_fix(phis, phi_ref, copy=False)

    return phis


