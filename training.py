import numpy as np
import time

import wandb
from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# torch.autograd.set_detect_anomaly(True)

import sys 
sys.path.append('./utils/') 
sys.path.append('./dataset/') 

# import custom libraries
import models as models
from dataset import Dataset_Bucketing
import energyflow_torch as efT
import utils

print('imports done')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ',device)



def training(params):

    ## a few additional parameters
    pt_log_scaling = False
    use_p3s = False     # using 3-vector notation
    scale_p3s = 1e-2   # scale input data points with this factor (p3s * scale_p3s)

    # Establish convention for real and fake labels during training
    real_label = 1.    # usually 1, but slightly lower might stabelize training "one-sided label smoothing"
    fake_label = 0.



    ########## Weights & Biases (wandb) initialisation
    if params.log_wandb:
        wandb.init(
            project=params.project_prefix+params.dataset_type+str(params.n_points),
            dir=params.wandb_dir,
            config=vars(params), 
            name='el'+str(params.equiv_layers_generator)+'_l'+str(params.latent)+'_'+str(params.rand)
            )
        # logging code with .py and .sh extension in specified dir (and all subdirs)  --> returns error when using multiprocessign / running multiple runs same time!
        # wandb.run.log_code(
            # "../", include_fn=lambda path: path.endswith(".py") or path.endswith(".sh")
        # )
        ## Capture a summary metric
        wandb.define_metric("w_dist_ms", summary="min")
    if params.log_comet:
        experiment = Experiment(project_name=params.project_prefix+params.dataset_type+str(params.n_points))
        experiment.set_name('el'+str(params.equiv_layers_generator)+'_l'+str(params.latent)+'_'+str(params.rand))
        experiment.log_parameters(vars(params))


    ##################################################      
    ##################################################
    ##################################################
    f_train, f_test, f_val, f_kde, norm_means, norm_stds, mins, maxs = utils.get_dataset(vars(params))


    # returns tensor in pytorch convention: batch, features, particles

    # bucketing
    dataset = Dataset_Bucketing(f_train, params.batch_size_max)
    dataloader = DataLoader(dataset, batch_size=None)
    len_iter_perEp = len(dataset)


    print(len_iter_perEp)


    ##################################################
    ##################################################

    # define model, loss

    ## arguments for network
    args = {'latent': params.latent,
            'latent_local': params.latent_local,
            'hid_d': params.hid_d,
            'feats': params.feats,
            'equiv_layers_generator': params.equiv_layers_generator,
            'equiv_layers_discriminator': params.equiv_layers_discriminator,
            'return_latent_space': False,
            }

    print('params:')
    print(vars(params))
    print('args:')
    print(args)

#######################    
#### LOAD NETWORKS ####    

    C, G = utils.get_model(model_name=params.model_name, args=args, latent=params.latent)  # load classifier C and generator G
            
    # log model gradients
    if params.log_wandb:
        wandb.watch(G, log_freq=1000)
        wandb.watch(C, log_freq=1000)
            
    
### OPTIM

    # reset iterations
    iteration = 0
    epoch = 0
    best_epoch = 0
    best_epoch_fpnd = 0
    best_w_dist_ms = 0
    test_w_dist_ms = 0
    best_fpnd = 0
    mean_errD, mean_loss_BCE = 0, 0

    iteration_list = []
    loss_tot_list = []
    out_ms_max_list = []


    ## GAN SPECIFIC STUFF
    criterion_BCE = nn.BCEWithLogitsLoss()  # when no sigmoid in last layer

    print('model initiated')

    # Model size
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Generator parameters: ', count_parameters(G))
    print('Discriminator parameters: ', count_parameters(C))


    ##################################################
    ##################################################

    optimizer_G = optim.Adam(G.parameters(), lr=params.lr, betas=(params.beta1, 0.999), eps=1e-14)
    optimizer_C = optim.Adam(C.parameters(), lr=params.lr_C, betas=(params.beta1, 0.999), eps=1e-14)

    ##################################################
    ##################################################

###### ACTUAL TRAINING LOOP ####

    # prediction
    G.train()
    C.train()

    #for epoch in range(params.epochs):
    ep_start = time.time()
    break_out = False

    len_iter_perEp = len(dataset)
    print(len_iter_perEp)    
    for _, sample_batch in enumerate(dataloader):
        iteration += 1

    ##### EPOCH TRACKING & EVALUATION LOOP (since the custom dataloader doesn't stop)
        if iteration % int(len_iter_perEp) == 0:   # now epoch = half epoch (not any more)
            print('done epoch no.: {}'.format(epoch+1))

            # validation
            w_dist_ms = utils.validation_mean_ms(G, f_val, f_kde, params.latent, params.latent_local, norm_means, norm_stds, params.norm_sigma, normalize_points=params.normalize_points, set_min_pt=True, min_pt = mins[0], return_latent_space = args['return_latent_space'], runs=10, center_gen=params.center_gen)
            # fpnd = utils.validation_FPND(G, f_test, f_kde, params.latent, params.latent_local, norm_means, norm_stds, params.norm_sigma, normalize_points=params.normalize_points, set_min_pt=True, min_pt = mins[0], return_latent_space = args['return_latent_space'], jet_type_fpnd=params.jet_type_fpnd, center_gen=params.center_gen)
            if params.log_wandb:
                wandb.log({"w_dist_ms": w_dist_ms,
                        "epoch": epoch,
                        # "FPND": fpnd,
                        }, step=iteration)
            if params.log_comet:
                experiment.log_metrics({"w_dist_ms": w_dist_ms,
                        "epoch": epoch,
                        # "FPND": fpnd,
                        }, step=iteration)
            print('w_dist_ms = {:.4f}'.format(w_dist_ms))
            if epoch == 0:
                best_w_dist_ms = w_dist_ms
                # best_fpnd = fpnd
                test_w_dist_ms = utils.validation_mean_ms(G, f_test, f_kde, params.latent, params.latent_local, norm_means, norm_stds, params.norm_sigma, normalize_points=params.normalize_points, set_min_pt=True, min_pt = mins[0], return_latent_space = args['return_latent_space'], runs=10, center_gen=params.center_gen)
                utils.save_model(G, optimizer_G, C, optimizer_C, fname=params.save_file_name+'_best_model', folder=params.save_folder)
                # utils.save_model(G, optimizer_G, C, optimizer_C, fname=params.save_file_name+'_best_model_fpnd', folder=params.save_folder)
                print('NEW BEST MODEL SAVED \n\n\n\n\n\n')
            else:  # save best model & calc w_dist on test set
                if w_dist_ms < best_w_dist_ms:
                    best_w_dist_ms = w_dist_ms
                    best_epoch = epoch  
                    utils.save_model(G, optimizer_G, C, optimizer_C, fname=params.save_file_name+'_best_model', folder=params.save_folder)
                    test_w_dist_ms = utils.validation_mean_ms(G, f_test, f_kde, params.latent, params.latent_local, norm_means, norm_stds, params.norm_sigma, normalize_points=params.normalize_points, set_min_pt=True, min_pt = mins[0], return_latent_space = args['return_latent_space'], runs=10, center_gen=params.center_gen)
                    if params.log_wandb:
                        plot = utils.plot_overview(G, f_test, f_kde, params.latent, params.latent_local, norm_means, norm_stds, params.norm_sigma, normalize_points=params.normalize_points, set_min_pt=True, min_pt = mins[0], return_latent_space = args['return_latent_space'], center_gen=params.center_gen)
                        wandb.log({"best_model_w_dist": wandb.Image(plot)}, step=iteration)
                    print('NEW BEST MODEL SAVED with test_w_dist_ms = {:.4f} in epoch {} \n\n\n\n\n\n'.format(test_w_dist_ms, best_epoch))
                    if params.log_wandb:
                        wandb.log({"best_w_dist_ms": best_w_dist_ms,
                                "test_w_dist_ms": test_w_dist_ms,
                                "best_epoch": best_epoch,
                                }, step=iteration)
                    if params.log_comet:
                        experiment.log_metrics({"best_w_dist_ms": best_w_dist_ms,
                                "test_w_dist_ms": test_w_dist_ms,
                                "best_epoch": best_epoch,
                                }, step=iteration)
                # if fpnd < best_fpnd:
                #     best_fpnd = fpnd
                #     best_epoch_fpnd = epoch  
                #     utils.save_model(G, optimizer_g, C, optimizer_C, fname=params.save_file_name+'_best_model_fpnd', folder=params.save_folder)
                #     if params.log_wandb:
                #         plot = utils.plot_overview(D, f_test, f_kde, params.latent, params.latent_local, norm_means, norm_stds, params.norm_sigma, normalize_points=params.normalize_points, set_min_pt=True, min_pt = mins[0], return_latent_space = args['return_latent_space'], center_gen=params.center_gen)
                #         wandb.log({"best_model_fpnd": wandb.Image(plot)}, step=iteration)
                #     print('NEW BEST MODEL SAVED based on fpnd = {}'.format(fpnd))
                #     if params.log_wandb:
                #         wandb.log({"best_fpnd": best_fpnd,
                #                 "best_epoch_fpnd": best_epoch_fpnd,
                #                 }, step=iteration)
                #     if params.log_comet:
                #         experiment.log_metrics({"best_fpnd": best_fpnd,
                #                 "best_epoch_fpnd": best_epoch_fpnd,
                #                 }, step=iteration)

            # # basic learning rate reduction schedule
            # if epoch == reduce_lr_after_Xepochs:
            #     lr, lr_C, beta1 = 1e-5, 1e-5, 0.9
            #     optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999), eps=1e-14)  # optimizes only decoder
            #     optimizer_C = optim.Adam(C.parameters(), lr=lr_C, betas=(0.9, 0.999), eps=1e-14)  # optimizes only decoder
            #     print('learning rate reduced to {}'.format(lr))

            epoch_time = (time.time() - ep_start)
            print('time of last epoch: ', epoch_time)
            if params.log_wandb:
                wandb.log({"epoch_time": epoch_time}, step=iteration)
            ep_start = time.time()  # reset epoch timer
            epoch += 1
            if epoch == params.epochs:    # stopping condition
                break

        n_points = sample_batch.size(1)

        # normalize dataset - assumes tensor in pytorch convention: batch, features, particles
        if (params.normalize_points == True) or (params.normalize_points_forDiscrOnly == True):
            #sample_batch = utils.normalize_tensor_logpt(sample_batch, mean=norm_means, std=norm_stds, sigma=params.norm_sigma)
            sample_batch = utils.normalize_tensor(sample_batch, mean=norm_means, std=norm_stds, sigma=params.norm_sigma)

        # move batch to GPU
        data = sample_batch.float().to(device)
        batch_size = data.size(0)

        if pt_log_scaling == True:
            data[:,0,:] = torch.log(data[:,0,:])

        # turn data into 3-vectors (px,py,pz)
        if use_p3s:
            data = efT.torch_p3s_from_ptyphi(data.permute(0,2,1)).permute(0,2,1) * scale_p3s




####### DISCRIMINATOR TRAINING
        label = torch.full((batch_size,1), real_label, dtype=torch.float, device=device)
        for _ in range(1): 
            C.train()
            G.eval()
            optimizer_C.zero_grad()
            C.zero_grad()

            # fake = encoded/decoded data
            z_global = utils.get_global_noise(batch_size, params.latent, device=device)
            z_local = utils.get_local_noise(batch_size, n_points, params.latent_local, device=device)
            out = G(z_global, z_local)

            # normalize dataset for discr. - assumes tensor in pytorch convention: batch, features, particles
            if params.normalize_points_forDiscrOnly == True:
                #sample_batch = utils.normalize_tensor_logpt(sample_batch, mean=norm_means, std=norm_stds, sigma=params.norm_sigma)
                out = utils.normalize_tensor(out, mean=norm_means, std=norm_stds, sigma=params.norm_sigma)
            

            if params.GAN_type == 'LSGAN':
                ### LSGAN (https://agustinus.kristia.de/techblog/2017/03/02/least-squares-gan/)
                discr_out_real = C(data)
                discr_out_fake = C(out)
                errD = 0.5 * (torch.mean((discr_out_real - real_label)**2) + torch.mean((discr_out_fake - fake_label)**2))
            else:
                # real = data    
                label_real = torch.full((batch_size,1), real_label, dtype=torch.float, device=device)
                label_fake = torch.full((batch_size,1), fake_label, dtype=torch.float, device=device)
                # concat real and fake
                label_cat = torch.cat((label_real, label_fake), dim=0)
                out_cat = torch.cat((data, out), dim=0)
                discr_out_cat = C(out_cat)#.view(-1)
                errD = criterion_BCE(discr_out_cat, label_cat)

            mean_errD += errD / params.log_interval # running loss average
            # Add the gradients from the all-real and all-fake batches
            #errD = errD_real + errD_fake
            # Update D
            errD.backward()
            optimizer_C.step()





####### GENERATOR TRAINING ###
        # zero grads
        C.eval()
        G.train()
        optimizer_G.zero_grad()
        G.zero_grad()

        z_global = utils.get_global_noise(batch_size, params.latent, device=device)
        z_local = utils.get_local_noise(batch_size, n_points, params.latent_local, device=device)   #shape: [batch, latent, points]
        out = G(z_global, z_local)

        # normalize dataset for discr. - assumes tensor in pytorch convention: batch, features, particles
        if params.normalize_points_forDiscrOnly == True:
            #sample_batch = utils.normalize_tensor_logpt(sample_batch, mean=norm_means, std=norm_stds, sigma=params.norm_sigma)
            out = utils.normalize_tensor(out, mean=norm_means, std=norm_stds, sigma=params.norm_sigma)

    ### DISCRIMINATOR LOSS - GAN LOSS
        if params.GAN_type == 'LSGAN':
            ### LSGAN (https://agustinus.kristia.de/techblog/2017/03/02/least-squares-gan/)
            discr_out= C(out)
            loss_BCE = 0.5 * torch.mean((discr_out - real_label)**2)
        else:
            label.fill_(real_label)    # real label = 1, fake label = 0
            discr_out = C(out)#.view(-1)
            loss_BCE = criterion_BCE(discr_out, label)


## CALC TOTAL LOSS  
        mean_loss_BCE += loss_BCE / params.log_interval # running loss average
        loss_tot = loss_BCE
        # loss backwards & steps
        loss_tot.backward()
        optimizer_G.step()


# stop training if model gets 'nan'
        #if (np.isnan(loss_reco.item()) == True) or (torch.isnan(out.max()) == True):
        for p in G.parameters():
            if torch.any(torch.isnan(p)) == True:
                break_out = True
        if break_out:
            print('nan detected')
            break_out=False
            break


        # online logging
        if iteration % params.log_interval == 0:
            if params.log_wandb:
                wandb.log({"discr_loss": mean_errD, 
                        "gen_loss": mean_loss_BCE,
                        "iteration": iteration,
                        }, step=iteration)
            if params.log_comet:
                experiment.log_metrics({"discr_loss": mean_errD, 
                        "gen_loss": mean_loss_BCE,
                        "iteration": iteration,
                        }, step=iteration)
            mean_errD, mean_loss_BCE = 0, 0
            
    # print & save current losses
        if iteration % params.save_interval == 0:

            # save model in a temporary file
            utils.save_model(G, optimizer_G, C, optimizer_C, fname=params.save_file_name, folder=params.save_folder)

        #if epoch % params.save_interval == 0:
            print(epoch, iteration)
            print('this batch size: ', out.shape[0], ' and this n_points: ', out.shape[2])
            print('total loss: ', loss_tot.item())
            print('errD loss during discriminator training: ', errD.item())
            print('loss_BCE GAN loss during enc/dec training: ', loss_BCE.item())
            iteration_list.append(iteration)
            loss_tot_list.append(loss_tot.item())

            # if enable_scheduler == True:
                # print('current lr: {:.3e} '.format(optimizer_E.param_groups[0]['lr']))

            # print minimum pt of batch - assumes (batch, feats, n_points)
            if (params.normalize_points == True) or (params.normalize_points_forDiscrOnly == True):
                #out = utils.inverse_normalize_tensor_logpt(out.detach(), mean=norm_means, std=norm_stds, sigma=params.norm_sigma)
                out = utils.inverse_normalize_tensor(out.detach(), mean=norm_means, std=norm_stds, sigma=params.norm_sigma)
                print('min pt value in batch: ',out[:,0,:].min())
                print('max pt value in batch: ',out[:,0,:].max())
                print('max first pt value in batch: ',out[:,0,0].max())
                out = out.permute(0,2,1)
                out_ys = efT.jet_ys(out)
                print('max jet y: ', out_ys.max())
                out_phis = efT.jet_phis(out)
                print('max jet phi: ', out_phis.max())
                out_ms = efT.jet_masses(out)
                print('min jet pt: ', efT.jet_pts(out).min())
                print('max jet pt: ', efT.jet_pts(out).max())
                print('max jet mass: ', out_ms.max())
                out_ms_max_list.append(out_ms.max())
            else:
                out = out.detach()
                # out[:,0,:] = out[:,0,:] / pt_scaling
                print('min pt value in batch: ',out[:,0,:].min())
                print('max pt value in batch: ',out[:,0,:].max())
                print('max y value in batch: ',out[:,1,:].max())
                out = out.permute(0,2,1)
                out_ys = efT.jet_ys(out)
                print('max jet y: ', out_ys.max())
                out_ms = efT.jet_masses(out)
                print('min jet pt: ', efT.jet_pts(out).min())
                print('max jet pt: ', efT.jet_pts(out).max())
                print('max jet mass: ', out_ms.max())
                out_ms_max_list.append(out_ms.max())


    print('training done, final total loss: ', loss_tot.item())
    
    
    return [best_w_dist_ms, test_w_dist_ms, best_epoch, epoch_time]