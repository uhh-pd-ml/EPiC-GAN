# adapted from: https://github.com/jet-net/JetNet/blob/main/jetnet/evaluation/gen_metrics.py
from energyflow.emd import emds
import numpy as np
from scipy.stats import wasserstein_distance
import jetnet
from torch import Tensor
from typing import Union, Tuple
from tqdm import tqdm


rng = np.random.default_rng()


def FPND(jet_ary, lenght=25_000, batches=10, jet_type='q'):
    lst, i = [], 0
    for _ in range(batches):
        fpnd = jetnet.evaluation.fpnd(jet_ary[i:i+lenght], jet_type=jet_type)
        i += lenght
        lst.append(fpnd)
    return np.mean(lst), np.std(lst)


def w1m(
    jets1: Union[Tensor, np.ndarray],
    jets2: Union[Tensor, np.ndarray],
    num_eval_samples: int = 10000,
    num_batches: int = 5,
    return_std: bool = True,
):
    """
        adapted such that jet1 = real_jets, jet2 = fake_jets
        no more random choice, rather compairng the whole test sets to batches of fake data
    """
    assert len(jets1.shape) == 3 and len(jets2.shape) == 3, "input jets format is incorrect"

    if isinstance(jets1, Tensor):
        jets1 = jets1.cpu().detach().numpy()

    if isinstance(jets2, Tensor):
        jets2 = jets2.cpu().detach().numpy()

    # shuffling jets
    rand1 = np.random.permutation(len(jets1))
    rand2 = np.random.permutation(len(jets2))
    jets1 = jets1[rand1]
    jets2 = jets2[rand2]

    masses1 = jetnet.utils.jet_features(jets1)["mass"]
    masses2 = jetnet.utils.jet_features(jets2)["mass"]

    w1s = []
    i = 0

    for _ in range(num_batches):
        rand1 = [*range(0,len(jets1))]  # whole real dataset
        rand2 = [*range(i,i+num_eval_samples)]  # batches of the fake dataset
        i += num_eval_samples

        rand_sample1 = masses1[rand1]
        rand_sample2 = masses2[rand2]

        w1s.append(wasserstein_distance(rand_sample1, rand_sample2))

    return np.mean(w1s), np.std(w1s) if return_std else np.mean(w1s)



def w1p(
    jets1: Union[Tensor, np.ndarray],   # real_jets
    jets2: Union[Tensor, np.ndarray],   # fake_jets
    mask1: Union[Tensor, np.ndarray] = None,
    mask2: Union[Tensor, np.ndarray] = None,
    exclude_zeros: bool = True,
    num_particle_features: int = 0,
    num_eval_samples: int = 10000,
    num_batches: int = 5,
    average_over_features: bool = True,
    return_std: bool = True,
):
    """
        adapted such that jet1 = real_jets, jet2 = fake_jets
        no more random choice, rather compairng the whole test sets to batches of fake data
    """
    assert len(jets1.shape) == 3 and len(jets2.shape) == 3, "input jets format is incorrect"

    if num_particle_features <= 0:
        num_particle_features = jets1.shape[2]

    assert (
        num_particle_features <= jets1.shape[2]
    ), "more particle features requested than were inputted"
    assert (
        num_particle_features <= jets2.shape[2]
    ), "more particle features requested than were inputted"

    if mask1 is not None:
        # TODO: should be wrapped in try catch
        mask1 = mask1.reshape(jets1.shape[0], jets1.shape[1])
        mask1 = mask1.astype(bool)

    if mask2 is not None:
        # TODO: should be wrapped in try catch
        mask2 = mask2.reshape(jets2.shape[0], jets2.shape[2])
        mask2 = mask2.astype(bool)

    if isinstance(jets1, Tensor):
        jets1 = jets1.cpu().detach().numpy()

    if isinstance(jets2, Tensor):
        jets2 = jets2.cpu().detach().numpy()

    # shuffling jets
    rand1 = np.random.permutation(len(jets1))
    rand2 = np.random.permutation(len(jets2))
    jets1 = jets1[rand1]
    jets2 = jets2[rand2]

    if exclude_zeros:
        zeros1 = np.linalg.norm(jets1[:, :, :num_particle_features], axis=2) == 0
        mask1 = ~zeros1 if mask1 is None else mask1 * ~zeros1

        zeros2 = np.linalg.norm(jets2[:, :, :num_particle_features], axis=2) == 0
        mask2 = ~zeros2 if mask2 is None else mask2 * ~zeros2

    w1s = []
    i = 0

    for _ in range(num_batches):
        rand1 = [*range(0,len(jets1))]  # whole real dataset
        rand2 = [*range(i,i+num_eval_samples)]  # batches of the fake dataset
        i += num_eval_samples

        rand_sample1 = jets1[rand1]
        rand_sample2 = jets2[rand2]

        if mask1 is not None:
            parts1 = rand_sample1[:, :, :num_particle_features][mask1[rand1]]
        else:
            parts1 = rand_sample1[:, :, :num_particle_features].reshape(-1, num_particle_features)

        if mask2 is not None:
            parts2 = rand_sample2[:, :, :num_particle_features][mask2[rand2]]
        else:
            parts2 = rand_sample2[:, :, :num_particle_features].reshape(-1, num_particle_features)

        if parts1.shape[0] == 0 or parts2.shape[0] == 0:
            w1 = [np.inf, np.inf, np.inf]
        else:
            w1 = [
                wasserstein_distance(parts1[:, i], parts2[:, i])
                for i in range(num_particle_features)
            ]

        w1s.append(w1)

    means = np.mean(w1s, axis=0)
    stds = np.std(w1s, axis=0)

    if average_over_features:
        return np.mean(means), np.linalg.norm(stds) if return_std else np.mean(means)
    else:
        return means, stds if return_std else means


def w1efp(
    jets1: Union[Tensor, np.ndarray],  # real jets
    jets2: Union[Tensor, np.ndarray],  # fake jets
    use_particle_masses: bool = False,
    efpset_args: list = [("n==", 4), ("d==", 4), ("p==", 1)],
    num_eval_samples: int = 10000,
    num_batches: int = 5,
    average_over_efps: bool = True,
    return_std: bool = True,
    efp_jobs: int = None,
):
    """
        adapted such that jet1 = real_jets, jet2 = fake_jets
        no more random choice, rather compairng the whole test sets to batches of fake data
    """

    if isinstance(jets1, Tensor):
        jets1 = jets1.cpu().detach().numpy()

    if isinstance(jets2, Tensor):
        jets2 = jets2.cpu().detach().numpy()

    # shuffling jets
    rand1 = np.random.permutation(len(jets1))
    rand2 = np.random.permutation(len(jets2))
    jets1 = jets1[rand1]
    jets2 = jets2[rand2]

    assert len(jets1.shape) == 3 and len(jets2.shape) == 3, "input jets format is incorrect"
    assert (jets1.shape[2] - int(use_particle_masses) >= 3) and (
        jets1.shape[2] - int(use_particle_masses) >= 3
    ), "particle feature format is incorrect"

    efps1 = jetnet.utils.efps(
        jets1, use_particle_masses=use_particle_masses, efpset_args=efpset_args, efp_jobs=efp_jobs
    )
    efps2 = jetnet.utils.efps(
        jets2, use_particle_masses=use_particle_masses, efpset_args=efpset_args, efp_jobs=efp_jobs
    )
    num_efps = efps1.shape[1]

    w1s = []
    i = 0

    for _ in range(num_batches):
        rand1 = [*range(0,len(jets1))]  # whole real dataset
        rand2 = [*range(i,i+num_eval_samples)]  # batches of the fake dataset
        i += num_eval_samples

        rand_sample1 = efps1[rand1]
        rand_sample2 = efps2[rand2]

        w1 = [wasserstein_distance(rand_sample1[:, i], rand_sample2[:, i]) for i in range(num_efps)]
        w1s.append(w1)

    means = np.mean(w1s, axis=0)
    stds = np.std(w1s, axis=0)

    if average_over_efps:
        return np.mean(means), np.linalg.norm(stds) if return_std else np.mean(means)
    else:
        return means, stds if return_std else means





def _optional_tqdm(iter_obj, use_tqdm, total=None, desc=None):
    if use_tqdm:
        return tqdm(iter_obj, total=total, desc=desc)
    else:
        return iter_obj


def cov_mmd(
    real_jets: Union[Tensor, np.ndarray],
    gen_jets: Union[Tensor, np.ndarray],
    num_eval_samples: int = 100,
    num_batches: int = 10,
    use_tqdm: bool = True,
) -> Tuple[float, float]:
    """
    Calculate coverage and MMD between real and generated jets,
    using the Energy Mover's Distance as the distance metric.
    Args:
        real_jets (Union[Tensor, np.ndarray]): Tensor or array of jets, of shape
          ``[num_jets, num_particles, num_features]`` with features in order ``[eta, phi, pt]``
        gen_jets (Union[Tensor, np.ndarray]): tensor or array of generated jets,
          same format as real_jets.
        num_eval_samples (int): number of jets out of the real and gen jets each between which to
          evaluate COV and MMD. Defaults to 100.
        num_batches (int): number of different batches to calculate COV and MMD and average over.
          Defaults to 100.
        use_tqdm (bool): use tqdm bar while calculating over ``num_batches`` batches.
          Defaults to True.
    Returns:
        Tuple[float, float]:
        - **float**: coverage, averaged over ``num_batches``.
        - **float**: MMD, averaged over ``num_batches``.
    """
    assert len(real_jets.shape) == 3 and len(gen_jets.shape) == 3, "input jets format is incorrect"
    assert (real_jets.shape[2] >= 3) and (
        gen_jets.shape[2] >= 3
    ), "particle feature format is incorrect"

    if isinstance(real_jets, Tensor):
        real_jets = real_jets.cpu().detach().numpy()

    if isinstance(gen_jets, Tensor):
        gen_jets = gen_jets.cpu().detach().numpy()

    assert np.all(real_jets[:, :, 2] >= 0) and np.all(
        gen_jets[:, :, 2] >= 0
    ), "particle pTs must all be >= 0 for EMD calculation"

    # convert from JetNet [eta, phi, pt] format to energyflow [pt, eta, phi]
    real_jets = real_jets[:, :, [2, 0, 1]]
    gen_jets = gen_jets[:, :, [2, 0, 1]]

    covs = []
    mmds = []

    for j in _optional_tqdm(
        range(num_batches), use_tqdm, desc=f"Calculating cov and mmd over {num_batches} batches"
    ):
        real_rand = rng.choice(len(real_jets), size=num_eval_samples)
        gen_rand = rng.choice(len(gen_jets), size=num_eval_samples)

        real_rand_sample = real_jets[real_rand]
        gen_rand_sample = gen_jets[gen_rand]

        # 2D array of emds, with shape (len(gen_rand_sample), len(real_rand_sample))
        dists = emds(gen_rand_sample, real_rand_sample)

        # for MMD, for each gen jet find the closest real jet and average EMDs
        mmds.append(np.mean(np.min(dists, axis=0)))

        # for coverage, for each real jet find the closest gen jet
        # and get the number of unique matchings
        covs.append(np.unique(np.argmin(dists, axis=1)).size / num_eval_samples)

    return np.mean(covs), np.std(covs), np.mean(mmds), np.std(mmds)