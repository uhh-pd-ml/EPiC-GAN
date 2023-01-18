import os
# from tokenize import Double
import torch
import numpy as np
import energyflow as ef
import sys
from scipy.stats import wasserstein_distance as w_dist
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import jetnet
matplotlib.use('Agg')  # to supress output

import energyflow_torch as efT
import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





# fixed centering of the jets
def center_jets(data):   # assumse [batch, particles, features=[pt,y,phi])
    etas = jet_etas(data)  # pseudorapdityt
    phis = jet_phis(data)  # azimuthal angle
    etas = etas[:,np.newaxis].repeat(repeats=data.shape[1], axis=1)
    phis = phis[:,np.newaxis].repeat(repeats=data.shape[1], axis=1)
    mask = data[...,0] > 0   # mask all particles with nonzero pt
    data[mask,1] -= etas[mask]
    data[mask,2] -= phis[mask]
    return data

# fixed centering of the jets
def center_jets_tensor(data):   # assumse [batch, particles, features=[pt,y,phi])
    etas = efT.jet_etas(data)  # pseudorapdityt
    phis = efT.jet_phis(data)  # azimuthal angle
    etas = etas[:,np.newaxis].expand(-1,data.shape[1])
    phis = phis[:,np.newaxis].expand(-1,data.shape[1])
    mask = data[...,0] > 0   # mask all particles with nonzero pt
    data[...,1][mask] -= etas[mask]   # there is a bug here when calculating gradients
    data[...,2][mask] -= phis[mask]  
    return data


    


### ARRAY HELPER FUNCTIONS ###

# for pt ordering of points
def order_ary(ary, order_dim = 0): # assumse array to be in shape [events, points, features]
    out_list = []
    events, pints, dims = ary.shape
    sortmask = np.argsort(ary[:,:,order_dim], axis=-1)
    for i in range(dims):
        new = np.take_along_axis(ary[:,:,i], sortmask, axis=-1)[:,::-1] # order reversed
        out_list.append(new)
    ordered_ary = np.stack(out_list, axis=-1)
    return ordered_ary

# for pt ordering of points - order 2nd array same as first array
def order_ary_2(ary1, ary2, order_dim = 0):
    out_list1, out_list2 = [], []
    if ary1.shape != ary2.shape:
        print("ERROR ary shape don't agree")
    events, pints, dims = ary1.shape
    sortmask = np.argsort(ary1[:,:,order_dim], axis=-1)
    for i in range(dims):
        new1 = np.take_along_axis(ary1[:,:,i], sortmask, axis=-1)[:,::-1] # order reversed
        new2 = np.take_along_axis(ary2[:,:,i], sortmask, axis=-1)[:,::-1]
        out_list1.append(new1)
        out_list2.append(new2)
    ordered_ary1 = np.stack(out_list1, axis=-1)
    ordered_ary2 = np.stack(out_list2, axis=-1)
    return ordered_ary1, ordered_ary2



# for pt ordering of points with torch
def order_tensor(tensor, order_dim = 0): # assumse array to be in shape [events, points, feats]
    events, points, dims = tensor.size()
    sortmask = torch.argsort(tensor[:,:,order_dim].reshape(events,points,1), dim=1, descending=True).expand(-1,-1,dims)
    tensor = torch.gather(tensor, 1, sortmask)
    return tensor




# assumes in pytorch convention: batch, particles, feats
def normalize_tensor(tensor, mean, std, sigma=1):
    for i in range(len(mean)):
        tensor[...,i] = (tensor[...,i] - mean[i]) / (std[i]/sigma) 
    return tensor

def inverse_normalize_tensor(tensor, mean, std, sigma=1):
    for i in range(len(mean)):
        tensor[...,i] = (tensor[...,i] * (std[i]/sigma)) + mean[i]
    return tensor


# assumes in pytorch convention: batch, particles, feats
# assumes pt is first variable and uses (natural) log preprocesing for that instead of standardization
def normalize_tensor_logpt(tensor, mean, std, sigma=1):
    for i in range(len(mean)):
        if i == 0:
            tensor[...,i] = torch.log(tensor[...,i])
        else:
            tensor[...,i] = (tensor[...,i] - mean[i]) / (std[i]/sigma)
    return tensor

def inverse_normalize_tensor_logpt(tensor, mean, std, sigma=1):
    for i in range(len(mean)):
        if i == 0:
            tensor[...,i] = torch.exp(tensor[...,i])
        else:
            tensor[...,i] = (tensor[...,i] * (std[i]/sigma)) + mean[i]
    return tensor

def inverse_normalize_tensor_ONLYyphi(tensor, mean, std, sigma=1):
    for i in range(len(mean)):
        if i == 0:
            tensor[...,i] = tensor[...,i]
        else:
            tensor[...,i] = (tensor[...,i] * (std[i]/sigma)) + mean[i]
    return tensor


# assumes tensor in pytorch convention: batch, poinrt, feats
# mean, std ary in shape [feats, points]
def normalize_tensor_pointwise(tensor, mean, std):
    mean = mean[:,0:tensor.shape[1]]
    std = std[:,0:tensor.shape[1]]
    return (tensor - mean) / std

def inverse_normalize_tensor_pointwise(tensor, mean, std):
    mean = mean[:,0:tensor.shape[1]]
    std = std[:,0:tensor.shape[1]]
    return (tensor * std) + mean


# min max scale to [0,1]
# assumes in pytorch convention: batch, particles, feats
def minmaxscale_tensor(tensor, mins, maxs):
    for i in range(len(mins)):
        tensor[...,i] = (tensor[...,i] - mins[i]) / (maxs[i] - mins[i])
    return tensor

def inverse_minmaxscale_tensor(tensor, mins, maxs):
    for i in range(len(mins)):
        tensor[...,i] = (tensor[...,i] * (maxs[i] - mins[i])) + mins[i]
    return tensor



## CALC JET VARIABLES

# a few jet kinematic definitions

def jet_masses(jets_ary): # in format (jets, particles, features)
    jets_p4s = ef.p4s_from_ptyphims(jets_ary)
    masses = ef.ms_from_p4s(jets_p4s.sum(axis=1))
    #masses = np.asarray([ef.ms_from_p4s(ef.p4s_from_ptyphims(x).sum(axis=0)) for x in jets_ary])
    return masses

def jet_pts(jets_ary): # in format (jets, particles, features)
    jets_p4s = ef.p4s_from_ptyphims(jets_ary)
    pts = ef.pts_from_p4s(jets_p4s.sum(axis=1))
    #pts = np.asarray([ef.pts_from_p4s(ef.p4s_from_ptyphims(x).sum(axis=0)) for x in jets_ary])
    return pts

def jet_ys(jets_ary):
    jets_p4s = ef.p4s_from_ptyphims(jets_ary)
    ys = ef.ys_from_p4s(jets_p4s.sum(axis=1))
    #ys = np.asarray([ef.ys_from_p4s(ef.p4s_from_ptyphims(x).sum(axis=0)) for x in jets_ary])
    return ys

def jet_etas(jets_ary):
    jets_p4s = ef.p4s_from_ptyphims(jets_ary)
    etas = ef.etas_from_p4s(jets_p4s.sum(axis=1))
    #etas = np.asarray([ef.etas_from_p4s(ef.p4s_from_ptyphims(x).sum(axis=0)) for x in jets_ary])
    return etas

def jet_phis(jets_ary):
    jets_p4s = ef.p4s_from_ptyphims(jets_ary)
    phis = ef.phis_from_p4s(jets_p4s.sum(axis=1), phi_ref=0)
    #phis = np.asarray([ef.phis_from_p4s(ef.p4s_from_ptyphims(x).sum(axis=0)) for x in jets_ary])
    return phis

def jet_mults(jets_ary):   # how many particles per jet
    mults = np.count_nonzero(jets_ary[:,:,0], axis=1)
    return mults




def create_folder_ifNotExists(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("New directory is created!")



def savetxt(filepath, ary, header):
    if os.path.exists(filepath):
        with open(filepath, "ab") as f:
            np.savetxt(f, ary)
        print('appended to ', filepath)
    else:
        np.savetxt(filepath, ary, header=header)
        print('created and saved to ', filepath)




def get_local_noise(batch_size, n_points, latent_local, device='cuda'):

    # local noise
    gen_noise_local = torch.empty((batch_size, n_points, latent_local), device=device)
    gen_noise_local.normal_(mean=0.0, std=1.0)
    return gen_noise_local

def get_global_noise(batch_size, latent, device='cuda'):

    # local noise
    gen_noise_global = torch.empty((batch_size, latent), device=device)
    gen_noise_global.normal_(mean=0.0, std=1.0)
    return gen_noise_global





def save_model(G, optimizer_G, C, optimizer_C, fname='tmp',
              folder = './tmp_trainings/'):
    path = folder+fname+'.tar'
    torch.save({
                'decoder_state': G.state_dict(),
                'optimizer_D_state': optimizer_G.state_dict(),
                'discr_state': C.state_dict(),
                'optimizer_C_state': optimizer_C.state_dict(),
                }, path)
    print('model temporary saved in ', path)
    
def load_model(G, optimizer_G, C, optimizer_C, fname='tmp', folder = './tmp_trainings/'):
    path = folder+fname+'.tar'
    checkpoint = torch.load(path)
    G.load_state_dict(checkpoint['decoder_state'], strict=True)
    optimizer_G.load_state_dict(checkpoint['optimizer_D_state'])
    C.load_state_dict(checkpoint['discr_state'], strict=True)
    optimizer_C.load_state_dict(checkpoint['optimizer_C_state'])
    print('model temporary loaded from ', path)
    return G, optimizer_G, C, optimizer_C




# validation function including kde multiplicity sampling: default comparing 5 generated sets with 1 data validation set
def validation_mean_ms(D, f_test, f_kde, latent, latent_local, norm_means, norm_stds, norm_sigma, normalize_points=True, set_min_pt=True, min_pt = 0.00013, return_latent_space = False, runs=10, center_gen = True):
    # retuns numpy arrays
    if return_latent_space:
        data_ary, gen_ary, _ = evaluation_loop(D, f_test, f_kde, latent, latent_local, norm_means, norm_stds, norm_sigma, normalize_points=normalize_points, set_min_pt=set_min_pt, min_pt = min_pt, return_latent_space = return_latent_space, gen_size_same_as_data=False, center_gen=center_gen)
    else:
        data_ary, gen_ary = evaluation_loop(D, f_test, f_kde, latent, latent_local, norm_means, norm_stds, norm_sigma, normalize_points=normalize_points, set_min_pt=set_min_pt, min_pt = min_pt, return_latent_space = return_latent_space, gen_size_same_as_data=False, center_gen=center_gen)
    data_ms = jet_masses(data_ary)
    data_len = len(data_ms)
    i = 0
    w_dist_list = []
    for _ in range(runs):
        gen_ms = jet_masses(gen_ary[i:i+data_len])
        i += data_len
        w_dist_ms = w_dist(data_ms, gen_ms)
        w_dist_list.append(w_dist_ms)
    return np.array(w_dist_list).mean()


# validation function including kde multiplicity sampling: FPND
def validation_FPND(D, f_test, f_kde, latent, latent_local, norm_means, norm_stds, norm_sigma, normalize_points=True, set_min_pt=True, min_pt = 0.00013, return_latent_space = False, jet_type_fpnd='g', center_gen = True):
    # retuns numpy arrays
    if return_latent_space:
        _, gen_ary, _ = evaluation_loop(D, f_test, f_kde, latent, latent_local, norm_means, norm_stds, norm_sigma, normalize_points=normalize_points, set_min_pt=set_min_pt, min_pt = min_pt, return_latent_space = return_latent_space, gen_size_same_as_data=True ,center_gen=center_gen)
    else:
        _, gen_ary = evaluation_loop(D, f_test, f_kde, latent, latent_local, norm_means, norm_stds, norm_sigma, normalize_points=normalize_points, set_min_pt=set_min_pt, min_pt = min_pt, return_latent_space = return_latent_space, gen_size_same_as_data=True, center_gen=center_gen)
    gen_ary = gen_ary[:,:,[1,2,0]]
    fpnd_score = jetnet.evaluation.fpnd(gen_ary, jet_type=jet_type_fpnd)
    return fpnd_score


def evaluation_loop(D, f_test, f_kde, latent, latent_local, norm_means, norm_stds, norm_sigma, normalize_points=True, 
                    return_latent_space = False, gen_size_same_as_data = True,
                    set_min_pt=True, batch_size=512, total_gen=300_000,
                    min_pt = 0.00013, order_points_pt=True, center_gen = True, calibrate_pt=False):
    reco_list, latent_list = [], []
    # load test data 
    data_ary = np.load(f_test)  # (jets, feats, particles)
    n_points_max = data_ary.shape[1]
    D.eval()

    # load kde of particle multiplicity
    with open(f_kde, 'rb') as f:
        kde = pickle.load(f)

    # sample all latent variables
    kde_sample = kde.resample(total_gen).T
    sampled_points = np.rint(kde_sample)
    sampled_points = sampled_points[(sampled_points >=1) & (sampled_points <= n_points_max)]
    unique_pts, unique_frqs = np.unique(sampled_points, return_counts=True)

    for i in range(len(unique_pts)):
        current_point = unique_pts[i]
        current_frq = unique_frqs[i]
        countdown = current_frq
        while countdown > 0:
            if countdown > batch_size:
                current_batch_size = batch_size
            else:
                current_batch_size = countdown
            
            countdown -= current_batch_size
            n_points = int(current_point)
            
            z_local = get_local_noise(current_batch_size, n_points, latent_local).to(device)
            z_global = get_global_noise(current_batch_size, latent).to(device)
            if return_latent_space:
                reco_test, latent_tensor = D(z_global,z_local)
                reco_test, latent_tensor = reco_test.detach().cpu().numpy(), latent_tensor.detach().cpu().numpy()
                latent_list.append(latent_tensor)
            else:
                reco_test = D(z_global,z_local).detach().cpu().numpy()

            if normalize_points == True:
                reco_test = inverse_normalize_tensor(torch.tensor(reco_test), mean=norm_means, std=norm_stds, sigma=norm_sigma).numpy()
                
            ## zero padding 
            reco_test_noPad = reco_test.copy()
            reco_test = np.zeros((reco_test_noPad.shape[0], n_points_max, 3))   ## WATCH OUT: here the feature dimension is hardcoded
            reco_test[0:reco_test_noPad.shape[0], 0:reco_test_noPad.shape[1], 0:reco_test_noPad.shape[2]] = reco_test_noPad
            
            reco_list.append(reco_test)

    gen_ary = np.vstack(reco_list)
    if return_latent_space:
        latent_np = np.vstack(latent_list)

    # and shuffle+cut on generated dataset
    perm = np.random.permutation(len(gen_ary))
    gen_ary = gen_ary[perm]
    if gen_size_same_as_data:
        gen_ary = gen_ary[:len(data_ary)]  # cut on events
    if return_latent_space:
        latent_np = latent_np[perm]
        if gen_size_same_as_data:
            latent_np = latent_np[:len(data_ary)]

    # back to physics convention (jets, particles, features)
    # data_ary = np.moveaxis(data_ary, 2,1)  
    # gen_ary = np.moveaxis(gen_ary, 2,1)

    # set negative pt values to minimum positive data pt value
    if set_min_pt:
        gen_ary[...,0][(gen_ary[...,0] < min_pt) & (gen_ary[...,0] != 0.0)] = min_pt

    if calibrate_pt:
        gen_pts = jet_pts(gen_ary)
        gen_ary[...,0] = gen_ary[...,0] / gen_pts.reshape(-1,1)

        set_min_pt = True
        if set_min_pt:
            gen_ary[...,0][(gen_ary[...,0] < min_pt) & (gen_ary[...,0] != 0.0)] = min_pt

    # order particles by pt
    if order_points_pt:
        data_ary = order_ary(data_ary, order_dim = 0)
        gen_ary = order_ary(gen_ary, order_dim = 0)

    if center_gen:
        gen_ary = center_jets(gen_ary)

    if return_latent_space:
        return data_ary, gen_ary, latent_np   # return as batch, particles, feats]
    else: 
        return data_ary, gen_ary  # returns numpy arrays





# validation function including kde multiplicity sampling: default comparing 5 generated sets with 1 data validation set
def plot_overview(D, f_test, f_kde, latent, latent_local, norm_means, norm_stds, norm_sigma, normalize_points=True, set_min_pt=True, min_pt = 0.00013, return_latent_space = False, center_gen = True):
    # retuns numpy arrays
    if return_latent_space:
        data_ary, gen_ary, _ = evaluation_loop(D, f_test, f_kde, latent, latent_local, norm_means, norm_stds, norm_sigma, normalize_points=normalize_points, set_min_pt=set_min_pt, min_pt = min_pt, return_latent_space = return_latent_space, center_gen=center_gen)
    else:
        data_ary, gen_ary = evaluation_loop(D, f_test, f_kde, latent, latent_local, norm_means, norm_stds, norm_sigma, normalize_points=normalize_points, set_min_pt=set_min_pt, min_pt = min_pt, return_latent_space = return_latent_space, center_gen=center_gen)

    data_test_ary = data_ary
    reco_test_ary = gen_ary

    # data variables
    data_ms = jet_masses(data_test_ary)
    data_pts = jet_pts(data_test_ary)
    data_mults = jet_mults(data_test_ary)
    # reco variables
    reco_ms = jet_masses(reco_test_ary)
    reco_pts = jet_pts(reco_test_ary)
    reco_mults = jet_mults(reco_test_ary)
    # p4s
    data_test_ary_p4s = efT.torch_p4s_from_ptyphi(torch.tensor(data_test_ary)).numpy()
    reco_test_ary_p4s = efT.torch_p4s_from_ptyphi(torch.tensor(reco_test_ary)).numpy()


    # overview plots
    density = 0
    n_points_cut = data_test_ary.shape[1]
    generator_name = 'EPiC-GAN'
    dataset_name = 'JetNet'+str(n_points_cut)
    color_list = ['grey', 'crimson', 'royalblue']
    fig = plt.figure(figsize=(18, 16), facecolor='white')
    gs = GridSpec(3,3)


    # particle pt 
    ax = fig.add_subplot(gs[0])
    x_min, x_max = np.array([data_test_ary[:,:,0].flatten().min(), reco_test_ary[:,:,0].flatten().min()]).min(), np.array([data_test_ary[:,:,0].flatten().max(), reco_test_ary[:,:,0].flatten().max()]).max()
    x_min, x_max = 0, 1.
    hist1 = ax.hist(data_test_ary[:,:,0][data_test_ary[:,:,0] != 0].flatten(), bins=100, label=dataset_name, histtype='stepfilled', range=[x_min,x_max], alpha=0.5, color=color_list[0])#, range=[-100,600])
    hist2 = ax.hist(reco_test_ary[:,:,0][reco_test_ary[:,:,0] != 0].flatten(), bins=100, label=generator_name, histtype='step', range=[x_min,x_max], lw=4, color=color_list[1])#, range=[-100,600])
    ax.legend(loc='upper right', fontsize=28,edgecolor='none')
    ax.set_xlabel(r'relative particle $p_\mathrm{T}^\mathrm{rel}$', fontsize=24)
    ax.set_ylabel('particles', fontsize=24)
    ax.set_yscale('log')
    ax.tick_params(labelsize=20)
    #plt.xscale('log')

    # particle rap
    ax = fig.add_subplot(gs[1])
    x_min, x_max = np.array([data_test_ary[:,:,1].flatten().min(), reco_test_ary[:,:,1].flatten().min()]).min(), np.array([data_test_ary[:,:,1].flatten().max(), reco_test_ary[:,:,1].flatten().max()]).max()
    x_min, x_max = -1.6,1.2
    hist1 = ax.hist(data_test_ary[:,:,1][data_test_ary[:,:,1] != 0].flatten(), bins=100, label=dataset_name, histtype='stepfilled', range=[x_min,x_max], alpha=0.5, color=color_list[0])
    hist2 = ax.hist(reco_test_ary[:,:,1][reco_test_ary[:,:,1] != 0].flatten(), bins=hist1[1], label=generator_name, histtype='step', range=[x_min,x_max], lw=4, color=color_list[1])
    ax.set_xlabel(r'particle pseudorapidity $\eta^\mathrm{rel}$', fontsize=24)
    ax.set_ylabel('particles', fontsize=24)
    ax.set_yscale('log')
    ax.set_xticks(np.linspace(x_min, x_max, 5))
    ax.tick_params(labelsize=20)

    # particle phi
    ax = fig.add_subplot(gs[2])
    x_min, x_max = np.array([data_test_ary[:,:,2].flatten().min(), reco_test_ary[:,:,2].flatten().min()]).min(), np.array([data_test_ary[:,:,2].flatten().max(), reco_test_ary[:,:,2].flatten().max()]).max()
    x_min, x_max = -.5,.5
    hist1 = ax.hist(data_test_ary[:,:,2][data_test_ary[:,:,2] != 0].flatten(), bins=100, label=dataset_name, histtype='stepfilled', range=[x_min,x_max], alpha=0.5, color=color_list[0])
    hist2 = ax.hist(reco_test_ary[:,:,2][reco_test_ary[:,:,2] != 0].flatten(), bins=hist1[1], label=generator_name, histtype='step', range=[x_min,x_max], lw=4, color=color_list[1])
    ax.set_xlabel(r'particle angle $\phi^\mathrm{rel}$', fontsize=24)
    ax.set_ylabel('particles', fontsize=24)
    ax.set_yscale('log')
    ax.set_ylim(1,)
    ax.tick_params(labelsize=20)


    n_points = [0,4,19]
    axes = [3,4,5]
    for j in range(3):
        i = n_points[j]
        ax = fig.add_subplot(gs[axes[j]])
        x_min, x_max = np.array([data_test_ary[:,i,0].flatten().min(), reco_test_ary[:,i,0].flatten().min()]).min(), np.array([data_test_ary[:,i,0].flatten().max(), reco_test_ary[:,i,0].flatten().max()]).max()
        hist1 = ax.hist(data_test_ary[:,i,0][data_test_ary_p4s[:,i,0] != 0].flatten(), bins=100, label=dataset_name, histtype='stepfilled', alpha=0.5, density=density, range=[x_min,x_max], color=color_list[0])
        hist2 = ax.hist(reco_test_ary[:,i,0][reco_test_ary_p4s[:,i,0] != 0].flatten(), bins=100, label=generator_name, histtype='step', density=density, range=[x_min,x_max], lw=4, color=color_list[1])#, range=[-100,600])
        if i == 0:
            ax.set_xlabel('{}'.format(i+1)+r'$^\mathrm{st}$ relative particle $p_\mathrm{T}^\mathrm{rel}$', fontsize=24)
        elif i == 1:
            ax.set_xlabel('{}'.format(i+1)+r'$^\mathrm{nd}$ relative particle $p_\mathrm{T}^\mathrm{rel}$', fontsize=24)
        elif i == 2:
            ax.set_xlabel('{}'.format(i+1)+r'$^\mathrm{rd}$ relative particle $p_\mathrm{T}^\mathrm{rel}$', fontsize=24)
        else:
            ax.set_xlabel('{}'.format(i+1)+r'$^\mathrm{th}$ relative particle $p_\mathrm{T}^\mathrm{rel}$', fontsize=24)
        ax.set_ylabel('particles', fontsize=24)
        ax.set_yscale('log')
        ax.set_xticks(ax.get_xticks()[1:][::2])   # hide every second x_tick, starting from the second
        ax.tick_params(labelsize=20)


    # jet mults
    ax = fig.add_subplot(gs[6])
    x_min, x_max = np.array([data_mults.min(), reco_mults.min()]).min(), np.array([data_mults.max(), reco_mults.max()]).max()
    b=x_max-x_min+1
    hist1 = ax.hist(data_mults, bins=b, label=dataset_name, histtype='stepfilled', alpha=0.5, range=[x_min,x_max], color=color_list[0])
    hist2 = ax.hist(reco_mults, bins=b, label=generator_name, histtype='step', lw=4, range=[x_min,x_max], color=color_list[1])
    ax.set_xlabel('particle multiplicity', fontsize=24)
    ax.set_yscale('log')
    ax.set_ylabel('jets', fontsize=24)
    ax.tick_params(labelsize=20)

    # jet mass
    ax = fig.add_subplot(gs[7])
    x_min, x_max = np.array([data_ms.min(), reco_ms.min()]).min(), np.array([data_ms.max(), reco_ms.max()]).max()
    x_min, x_max = 0, 0.3
    hist1 = ax.hist(data_ms, bins=100, label=dataset_name, histtype='stepfilled', range=[x_min,x_max], alpha=0.5, color=color_list[0])
    hist2 = ax.hist(reco_ms, bins=100, label=generator_name, histtype='step', range=[x_min,x_max], lw=4, color=color_list[1])
    ax.set_xlabel('relative jet mass', fontsize=24)
    ax.set_ylabel('jets', fontsize=24)
    ax.set_yscale('log')
    ax.tick_params(labelsize=20)

    #jet pt
    ax = fig.add_subplot(gs[8])
    x_min, x_max = np.array([data_pts.min(), reco_pts.min()]).min(), np.array([data_pts.max(), reco_pts.max()]).max()
    x_min, x_max = 0.6, 1.2
    hist1 = ax.hist(data_pts, bins=100, label=dataset_name, histtype='stepfilled', range=[x_min,x_max], alpha=0.5, color=color_list[0])#, range=[-10,100])
    hist2 = ax.hist(reco_pts, bins=100, label=generator_name, histtype='step', range=[x_min,x_max], lw=4, color=color_list[1])
    ax.set_xlabel(r'relative jet $p_\mathrm{T}^\mathrm{rel}$', fontsize=24)
    ax.set_ylabel('jets', fontsize=24)
    ax.set_ylim(1,)
    ax.set_yscale('log')
    ax.tick_params(labelsize=20)

    plt.tight_layout()

    return fig





def get_model(model_name, args ,latent=10, only_generator=False):

    if model_name == 'EPiC_GAN':
        if not only_generator:
            C = models.EPiC_discriminator(args).to(device)   # discriminator  (classifier)
        if latent == 0:
            print('using only local model')
            sys.exit('ERROR: no local model latent=0 implemented')
        else:
            G = models.EPiC_generator(args).to(device)   # generator
    else:
        sys.exit('ERROR: model unknown')

    if only_generator:
        return G
    else:
        return C, G



def get_dataset(params):
    n_points = params['n_points']
    norm_means, norm_stds, mins, maxs = [], [], [], []
    
    out_folder='./dataset/'
    if n_points == 30:
        if params['dataset_type'] == 'jetnet_gluon':
            outfile_prefix = 'gluon_jetnet30_'
        elif params['dataset_type'] == 'jetnet_quark':
            outfile_prefix = 'quark_jetnet30_'
        elif params['dataset_type'] == 'jetnet_top':
            outfile_prefix = 'top_jetnet30_'
        else:
            sys.exit('ERROR: DATASET TYPE NOT DEFINED')
        

    elif n_points == 150:
        if params['dataset_type'] == 'jetnet_gluon':
            outfile_prefix = 'gluon_jetnet150_'
        elif params['dataset_type'] == 'jetnet_quark':
            outfile_prefix = 'quark_jetnet150_'
        elif params['dataset_type'] == 'jetnet_top':
            outfile_prefix = 'top_jetnet150_'
        else:
            sys.exit('ERROR: DATASET TYPE NOT DEFINED')
    else:
        sys.exit('ERROR: NO DATASET FOR THIS NUMBER OF POINTS DEFINED')

    f_train = out_folder + outfile_prefix + '_train.npy'
    f_test = out_folder + outfile_prefix + '_test.npy'
    f_val = out_folder + outfile_prefix + '_val.npy'
    f_kde = out_folder + outfile_prefix + '_mults_kde.pkl'
    norm_means, norm_stds, mins, maxs = dataset_means_stds_mins_maxs(f_train)
    return f_train, f_test, f_val, f_kde, norm_means, norm_stds, mins, maxs




def dataset_means_stds_mins_maxs(f_train):
    data_train = np.load(f_train)    # assumes [batch, particles, feats]
    # mean and std considering (removing) zero padding
    mean_list = []
    std_list = []
    min_list = []
    max_list = []
    for i in range(3):
        data = data_train[:,:,i].flatten()
        data = data[data != 0.]
        mean_list.append(data.mean())
        std_list.append(data.std())
        min_list.append(data.min())
        max_list.append(data.max())
    return mean_list, std_list, min_list, max_list
