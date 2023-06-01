import argparse
import numpy as np
import time
import pickle

import torch

import sys 
sys.path.append('./utils/') 
import utils

print('imports done')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ',device)




def main():

    ###### COMMAND LINE ARGUMENTS
    # define parser
    parser = argparse.ArgumentParser(description='generation of particle jets via trained EPiC-GAN for JetNet30 / JetNet150 (gluon / light quarks / top jets)')

    # add possible argument

    ## OFTEN CHANGED
    parser.add_argument('--n_points', '-n', default=30, help='number of points to generate (either 30 or 150) for JetNet30 or JetNet150', type=int)
    parser.add_argument('--dataset_type', default='jetnet_top', help='define the dataset type, either jetnet_gluon, jetnet_quark, or jetnet_top', type=str )
    parser.add_argument('--events', '-e', default=100_000, type=int, help='number of events / number of jets to generate, should be at least 100k to ensure accurate particle multiplicity diversity')
    parser.add_argument('--batch_size', '-b', default=512, type=int, help='batch size')
    parser.add_argument('--output_folder', default='./', type=str, help='set output folder for generated jets in npy format')
    parser.add_argument('--outfile_name', default='EPiC_jets.npy', type=str, help='output filename for the .npy file containing the generated jets')

    # read parser into class
    params = parser.parse_args()

    ############# FIXED ARGUMENTS FROM TRAINING
    params.equiv_layers_generator = 6
    params.equiv_layers_discrmiminator = 3
    params.hid_d = 128 
    params.latent = 10
    params.latent_local = 3
    params.feats = 3
    params.norm_sigma = 5
    params.model_name = 'EPiC_GAN'


    ##############################


    # LOAD KDE of particle multiplicity distribution
    f_kde, norm_means, norm_stds, mins = utils.get_kde_etc(vars(params))

    # define model arguments and load model
        ## arguments for network
    args = {'latent': params.latent,
            'latent_local': params.latent_local,
            'hid_d': params.hid_d,
            'feats': params.feats,
            'equiv_layers_generator': params.equiv_layers_generator,
            'equiv_layers_discriminator': params.equiv_layers_discrmiminator,
            'return_latent_space': False,
            }


    G = utils.get_model(model_name=params.model_name, args=args, only_generator=True)  # load generator G
    G = load_model(G, params)
    
    ## GENERATE and SAVE jets
    start_time = time.time()

    gen_ary = generation(G, f_kde, params.latent, params.latent_local, norm_means, norm_stds, params.norm_sigma, normalize_points=True, 
                    set_min_pt=True, batch_size=params.batch_size, total_gen=params.events, n_points_max=params.n_points,
                    min_pt = mins[0], order_points_pt=True, center_gen = True, calibrate_pt=False)

    stop_time = time.time()
    print('generation done in {} seconds'.format(stop_time - start_time))
    print('time per jet: {} microseconds'.format((stop_time - start_time)/params.events * 1e6))
    print('(note that these timings are not comparable to the timing mentioned in the papaer, as here we are generating jets with a variable particle multiplicity)')

    np.save(params.output_folder+params.outfile_name, gen_ary)

    print('Output shape: {}. Order of features: [p_t, rapitiy eta, angle phi]. Done.'.format(gen_ary.shape))






def load_model(G, params, folder = './trained_models/'):
    path = folder+'EPiC_'+params.dataset_type+'_'+str(params.n_points)+'.tar'
    checkpoint = torch.load(path)
    G.load_state_dict(checkpoint['decoder_state'], strict=True)
    G.eval()
    return G



def generation(G, f_kde, latent, latent_local, norm_means, norm_stds, norm_sigma, normalize_points=True, 
                    set_min_pt=True, batch_size=512, total_gen=300_000, n_points_max=30,
                    min_pt = 0.00013, order_points_pt=True, center_gen = True, calibrate_pt=False):
    reco_list = []

    # load kde of particle multiplicity
    with open(f_kde, 'rb') as f:
        kde = pickle.load(f)

    # sample all latent variables
    kde_sample = kde.resample(total_gen+1000).T  # add 1000 events to compensate for removal of points outside the kde window
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
            
            z_local = utils.get_local_noise(current_batch_size, n_points, latent_local, device=device)
            z_global = utils.get_global_noise(current_batch_size, latent, device=device)
            with torch.no_grad():
                gen_tensor = G(z_global,z_local)

            if normalize_points == True:
                gen_tensor = utils.inverse_normalize_tensor(gen_tensor, mean=norm_means, std=norm_stds, sigma=norm_sigma)
            else:
                gen_tensor = gen_tensor.detach().cpu().numpy()

            if center_gen:
                gen_tensor = utils.center_jets_tensor(gen_tensor)

            # order particles by pt
            if order_points_pt:
                gen_tensor = utils.order_tensor(gen_tensor, order_dim = 0)

            gen_tensor = gen_tensor.detach().cpu().numpy()
                
            ## zero padding 
            gen_tensor_noPad = gen_tensor.copy()
            gen_tensor = np.zeros((gen_tensor_noPad.shape[0], n_points_max, 3))   ## WATCH OUT: here the feature dimension is hardcoded
            gen_tensor[0:gen_tensor_noPad.shape[0], 0:gen_tensor_noPad.shape[1], 0:gen_tensor_noPad.shape[2]] = gen_tensor_noPad
            
            reco_list.append(gen_tensor)

    gen_ary = np.vstack(reco_list)

    # and shuffle generated dataset and cut
    perm = np.random.permutation(len(gen_ary))
    gen_ary = gen_ary[perm]
    gen_ary = gen_ary[0:total_gen]

    # set negative pt values to minimum positive data pt value
    if set_min_pt:
        gen_ary[...,0][(gen_ary[...,0] < min_pt) & (gen_ary[...,0] != 0.0)] = min_pt

    if calibrate_pt:
        gen_pts = utils.jet_pts(gen_ary)
        gen_ary[...,0] = gen_ary[...,0] / gen_pts.reshape(-1,1)

        set_min_pt = True
        if set_min_pt:
            gen_ary[...,0][(gen_ary[...,0] < min_pt) & (gen_ary[...,0] != 0.0)] = min_pt

    # # order particles by pt
    # if order_points_pt:
    #     gen_ary = utils.order_ary(gen_ary, order_dim = 0)

    return gen_ary  # returns numpy arrays




if __name__ == "__main__":
    main()