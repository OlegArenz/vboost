from os import sys, path
sys.path.append(path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__))))))

"""
Script that runs the `frisk` and `baseball` models.  Saves inferences
for separate, model-specific plotting scripts.
"""

#########
# cli   #
#########
import argparse, os
parser = argparse.ArgumentParser(
    description="Run vboost, mcmc, and npvi on two hierarchical models")
parser.add_argument('-model', type=str, default='frisk', help="frisk|baseball|planarRobot|planarRobot4p_10")

parser.add_argument('-progressDir', type=str, default='progress_dir', help="progress_dir")

parser.add_argument('-vboost', action='store_true',
                    help='run vboost and save')
parser.add_argument('-vboost_nsamps',  type=int, default=100,
                    help='number of samples for gradient estimates')
parser.add_argument('-npvi',   action='store_true',
                    help='run nonparametric vi and save')
parser.add_argument('-rank',   type=int, default=0,
                    help='rank for vboost components')
parser.add_argument('-ncomp',  type=int, default=20,
                    help='number of components to use with vboost or NPVI')
parser.add_argument('-mcmc',   action='store_true',
                    help='run mcmc (NUTS) and save')
parser.add_argument('-mcmc_nsamps', type=int, default=5000,
                    help='number of MCMC steps to take')
parser.add_argument('-seed', type=int, default=12, help="random seed...")
args, _ = parser.parse_known_args()


############################
# shared model setup code  #
############################
import pandas as pd
from autograd import grad
import autograd.numpy as np
import autograd.numpy.random as npr
from vbproj import vi
from vbproj.gsmooth.opt import FilteredOptimization
from vbproj.gsmooth.smoothers import AdamFilter
from vbproj.models import baseball, frisk
from vbproj.vi.vboost.components import LRDComponent
import cPickle as pickle
from autograd.scipy.stats import multivariate_normal as normal_pdf
from time import time
# create output for fit VBoost model
args.output = args.model + args.progressDir
if not os.path.exists(args.output):
    os.makedirs(args.output)

# set up model function --- autograd-able
def make_model(model_name):
    if model_name == "baseball":
        # baseball model and data
        lnpdf_named = baseball.lnpdf
        lnpdf_flat  = baseball.lnpdf_flat
        lnpdft      = lambda z, t: np.squeeze(lnpdf_flat(z, t))
        lnpdf       = lambda z: np.squeeze(lnpdf_flat(z, 0))
        D           = baseball.D
        return lnpdf, D, None
    elif model_name == "frisk":
        lnpdf_tmp, unpack, D, sdf, pnames = frisk.make_model_funs(precinct_type=1)
        def lnpdf(th):
            lnpdf.counter += len(np.atleast_2d(th))
            return lnpdf_tmp(th)
        lnpdf.counter = 0
        return lnpdf, D, pnames
    elif model_name == "planarRobot_2":
        from experiments.lnpdfs.create_target_lnpfs import build_target_likelihood_planar_autograd
        lnpdf = build_target_likelihood_planar_autograd(2)[0]
        return lnpdf, 2, None
    elif model_name == "planarRobot_3":
        from experiments.lnpdfs.create_target_lnpfs import build_target_likelihood_planar_autograd
        lnpdf = build_target_likelihood_planar_autograd(3)[0]
        return lnpdf, 3, None
    elif model_name == "planarRobot_10":
        from experiments.lnpdfs.create_target_lnpfs import build_target_likelihood_planar_autograd
        lnpdf = build_target_likelihood_planar_autograd(10)[0]
        return lnpdf, 10, None
    elif model_name == "planarRobot4p_10":
        from experiments.lnpdfs.create_target_lnpfs import build_target_likelihood_planar_4p_autograd
        lnpdf = build_target_likelihood_planar_4p_autograd(10)[0]
        return lnpdf, 10, None
    elif model_name == "GMM_20":
        from experiments.lnpdfs.create_target_lnpfs import build_GMM_lnpdf_autograd
        [lnpdf, true_means, true_covs] = build_GMM_lnpdf_autograd(20,10)
        np.savez(args.output+'target_gmm.npz', true_means=true_means, true_covs=true_covs)
        return lnpdf, 20, None
    elif model_name == "GMM_2":
        from experiments.lnpdfs.create_target_lnpfs import build_GMM_lnpdf_autograd
        [lnpdf, true_means, true_covs] = build_GMM_lnpdf_autograd(2,10)
        np.savez(args.output+'target_gmm.npz', true_means=true_means, true_covs=true_covs)
        return lnpdf, 2, None
    elif model_name == "german_credit":
        from experiments.lnpdfs.create_target_lnpfs import build_german_credit_lnpdf
        lnpdf = build_german_credit_lnpdf(with_autograd=True)
        return lnpdf, 25, None
    elif model_name == "breast_cancer":
        from experiments.lnpdfs.create_target_lnpfs import build_breast_cancer_lnpdf
        lnpdf = build_breast_cancer_lnpdf(with_autograd=True)
        return lnpdf, 31, None
    elif model_name == "iono":
        from experiments.lnpdfs.create_target_lnpfs import build_GPR_iono_with_grad_lnpdf
        lnpdf_grad = build_GPR_iono_with_grad_lnpdf(remove_autograd=True)
        def lnpdf(theta):
            theta = np.atleast_2d(theta)
            lnpdf.counter += len(theta)
            output = []
            for t in theta:
                output.append(lnpdf_grad(t)[0])
            return np.array(output)
        lnpdf.counter = 0
        return lnpdf, 34, None

np.random.seed(args.seed)
lnpdf, D, _ = make_model(args.model)
# def lnpdf(theta):
#     if theta.ndim == 1:
#         lnpdf.counter+=1
#     else:
#         lnpdf.counter+=len(theta)
#     return lnpdf_tmp(theta)
# lnpdf.counter=0
#
# def dlnpdf(theta):
#     if theta.ndim == 1:
#         dlnpdf.counter+=1
#     else:
#         dlnpdf.counter+=len(theta)
#     return lnpdf_tmp(theta)[1]
# dlnpdf.counter=0

#
# print script args
#
if __name__=="__main__":
    print \
    """
    =========== mlm_main ==============

      model                    : {model}
      posterior dimension (D)  : {D}
      output dir               : {output_dir}

    """.format(model=args.model, D=D, output_dir=args.output)
    print args

# single component initialization
def init_single_component(rank=0, niter=2000, samples_for_init_comp=1024):
    infobj = vi.LowRankMvnBBVI(lnpdf, D, r=rank, lnpdf_is_vectorized=True)
    elbo_grad = grad(infobj.elbo_mc)
    def mc_grad_fun(lam, t):
        return -1.*elbo_grad(lam, n_samps=samples_for_init_comp)

    # initialize var params
    m0   = np.random.randn(D) * .01
    C0   = np.random.randn(D, infobj.r) * .001
    v0   = -3 * np.ones(D)
    lam0 = infobj.pack(m0, v0, C0)

    mc_opt = FilteredOptimization(
      mc_grad_fun,
      lam0.copy(),
      save_grads = False,
      grad_filter = AdamFilter(),
      fun = lambda lam, t: infobj.elbo_mc(lam, n_samps=100),
      callback=infobj.callback)
    mc_opt.run(num_iters=niter, step_size=.05)
    return mc_opt.params.copy(), infobj

def mfvi_init(niter=1000):
    mfvi = vi.DiagMvnBBVI(lnpdf, D, lnpdf_is_vectorized=True)
    elbo_grad = grad(mfvi.elbo_mc)
    def mc_grad_fun(lam, t):
        return -1.*elbo_grad(lam, n_samps=1024)

    lam = np.random.randn(mfvi.num_variational_params) * .01
    lam[D:] = -1.
    mc_opt = FilteredOptimization( mc_grad_fun, lam.copy(),
      save_grads  = False,
      grad_filter = AdamFilter(),
      fun         = lambda lam, t: mfvi.elbo_mc(lam, n_samps=1000),
      callback    = lambda th, t, g: mfvi.callback(th, t, g, n_samps=1000))
    mc_opt.run(num_iters=niter, step_size=.05)
    return mc_opt.params.copy()
#
#
# mfvi_params = mfvi_init()
# lrd_params  = init_single_component(rank=0)
# lrd_params  = init_single_component(rank=3)

###########################
# Variational Boosting    #
###########################

if args.vboost:
    timestamps = []
    timestamps.append(time())
    # initialize a single component of appropriate rank and cache
    initialization_iters = 500
    initialization_samples = D
    vi_params, infobj = init_single_component(rank=args.rank, niter=initialization_iters, samples_for_init_comp=initialization_samples)
    init_file = os.path.join(args.output, "initial_component-rank_%d.npy"%args.rank)
    lnpdf.counter - initialization_iters * initialization_samples
    np.save(init_file, vi_params)

    # initialize LRD component (works with MixtureVI ...)
    comp     = LRDComponent(D, rank=args.rank)
    m, d, C  = vi_params[:D], vi_params[D:(2*D)], vi_params[2*D:]
    comp.lam = comp.setter(vi_params.copy(), mean=m, v=d, C=C)

    # Variational Boosting Object (with initial component
    from vbproj.vi.vboost import mog_bbvi
    vbobj = vi.MixtureVI(lambda z, t: lnpdf(z),
                         D                     = D,
                         comp_list             = [(1., comp)],
                         fix_component_samples = True,
                         break_condition='percent')

    # iteratively add comps
    for k in xrange(args.ncomp):

        print "======== adding component %d ==============="%k

        # initialize new comp w/ weighted EM scheme
        (init_prob, init_comp) = \
            vbobj.fit_mvn_comp_iw_em(new_rank = comp.rank,
                                     num_samples=2000,
                                     importance_dist = 'gauss-mixture',
                                     use_max_sample=False)
        init_prob = np.max([init_prob, .5])

        true_counter = lnpdf.counter
        lnpdf.counter = 0
        # fit new component
        vbobj.fit_new_comp(init_comp = init_comp,
                           init_prob = init_prob,
                           max_iter  = 500,
                           step_size = .1,
                           num_new_component_samples   =10*D,
                           num_previous_mixture_samples=10*D,
                           fix_component_samples=True,
                           gradient_type="standard", #component_approx_static_rho",
                           break_condition='percent')
        lnpdf.counter = true_counter + lnpdf.counter /2 # we don't count the evaluations by the progress callback

        # after all components are added, tune the weights of each comp
        comp_list = mog_bbvi.fit_mixture_weights(vbobj.comp_list, vbobj.lnpdf,
                                                num_iters=1000, step_size=.25,
                                                num_samps_per_component=10*D,
                                                ax=None)
        vbobj.comp_list = comp_list

        timestamps.append(time())
        samples = vbobj.sample(2000)
        np.savez(os.path.join(args.output,"vboost_comp_%d.npz"  %  k), samples, timestamps, lnpdf.counter)
        # save output here
        vb_outfile = os.path.join(args.output,
                                  "vboost_comp_%d.pkl" % k)
        lam_list = [(p, c.lam) for p, c in vbobj.comp_list]
        with open(vb_outfile, 'wb') as f:
            pickle.dump(lam_list, f)

    ## save output here
    #vb_outfile = os.path.join(args.output, "vboost.pkl")
    #lam_list = [(p, c.lam) for p, c in vbobj.comp_list]
    #with open(vb_outfile, 'wb') as f:
    #    pickle.dump(lam_list, f)


#############################################
# Nonparametric variational inference code  #
#  --- save posterior parameters            #
#############################################

if args.npvi:

    init_with_mfvi = True
    if init_with_mfvi:
        mfvi_lam = mfvi_init()

        # initialize theta
        theta_mfvi = np.atleast_2d(np.concatenate([ mfvi_lam[:D],
                                                    [2*mfvi_lam[D:].mean()] ]))
        mu0        = vi.bbvi_npvi.mogsamples(args.ncomp, theta_mfvi)

        # create npvi object
        theta0 = np.column_stack([mu0, np.ones(args.ncomp)*theta_mfvi[0,-1]])

    else:
        theta0 = np.column_stack([10 * np.random.randn(args.ncomp, D),
                                  -2 * np.ones(args.ncomp)])

    # create initial theta and sample
    npvi = vi.NPVI(lnpdf, D=D)
    mu, s2, elbo_vals, theta = npvi.run(theta0.copy(), niter=1000, verbose=False, path=args.output)
    print elbo_vals

    # save output here
    npvi_outfile = os.path.join(args.output, "npvi_%d-comp.npz"%args.ncomp)
    np.savez(npvi_outfile, theta, mu, s2)


#########################################
# MCMC code --- save posterior samples  #
#########################################

if args.mcmc:

    import sampyl
    if args.model == "baseball":
        nuts = sampyl.NUTS(baseball.lnp,
                   start={'logit_phi'  : np.random.randn(1),
                          'log_kappa'  : np.random.randn(1),
                          'logit_theta': np.random.randn(D - 2) })

        # keep track of number of LL calls
        cum_ll_evals = np.zeros(args.mcmc_nsamps, dtype=np.int)
        def callback(i):
            cum_ll_evals[i] = lnpdf.called
            if i % 500 == 0:
                print "total lnpdf calls", lnpdf.called

        lnpdf.called = 0
        chain = nuts.sample(args.mcmc_nsamps, burn=0, callback=callback)
        lnpdf.called = 0
        # compute log like of each sample
        lls   = np.array([ baseball.lnp(*c) for c in chain ])
        # save chain
        nuts_dict = {'chain': chain, 'lls': lls, 'cum_ll_evals': cum_ll_evals}

    elif args.model == "frisk":
        nuts = sampyl.NUTS(lnpdf, start={"th": np.random.randn(D) * .1})
        chain = []
        n_fevals = []
        start = time()
        timestamps = []
        n_samps_per_iter = 1000
        while len(chain) * n_samps_per_iter < args.mcmc_nsamps:
            if args.mcmc_nsamps > len(chain) + n_samps_per_iter:
                chain.append(np.vstack([samp[0] for samp in nuts.sample(n_samps_per_iter, burn=0)]))
            else:
                chain.append(np.vstack([samp[0] for samp in nuts.sample(args.mcmc_nsamps - len(chain) * n_samps_per_iter, burn=0)]))
            timestamps.append(time() - start)
            n_fevals.append(lnpdf.counter)
            np.savez(args.model + args.progressDir + "NUTS_" + str(len(chain)*n_samps_per_iter ) + "of" + str(args.mcmc_nsamps) + ".npz",
                     samples=np.array(chain), fevals=np.array(n_fevals),
                     timestamps=np.array(timestamps))
        np.savez(args.model + args.progressDir + "NUTS_" + str(args.mcmc_nsamps) + "processed_data.npz",
                 samples=np.array(chain),
                 fevals=np.array(n_fevals), timestamps=np.array(timestamps))
        nuts_dict = {'chain': chain}

    elif args.model == "planarRobot_10":
        conf_likelihood_var = 4e-2 * np.ones(10)
        conf_likelihood_var[0] = 1
        from scipy.stats import multivariate_normal
        x0 = multivariate_normal(np.zeros(10), conf_likelihood_var * np.eye(10)).rvs(1)
        nuts = sampyl.NUTS(lnpdf, start={"theta": x0})
        chain = []
        n_fevals = []
        start = time()
        timestamps = []
        n_samps_per_iter = 100
        while len(chain) * n_samps_per_iter < args.mcmc_nsamps:
            if args.mcmc_nsamps > len(chain) + n_samps_per_iter:
                chain.append(np.vstack([samp[0] for samp in nuts.sample(n_samps_per_iter, burn=0)]))
            else:
                chain.append(np.vstack([samp[0] for samp in nuts.sample(args.mcmc_nsamps - len(chain) * n_samps_per_iter, burn=0)]))
            timestamps.append(time() - start)
            n_fevals.append(lnpdf.counter)
            np.savez(args.model + args.progressDir + "NUTS_" + str(len(chain)*n_samps_per_iter ) + "of" + str(args.mcmc_nsamps) + ".npz",
                     samples=np.array(chain), fevals=np.array(n_fevals),
                     timestamps=np.array(timestamps))
        np.savez(args.model + args.progressDir + "NUTS_" + str(args.mcmc_nsamps) + "processed_data.npz",
                 samples=np.array(chain),
                 fevals=np.array(n_fevals), timestamps=np.array(timestamps))
        #   lls       = np.array([ lnpdf_grad(*c) for c in chain ])
        nuts_dict = {'chain': chain}
        nuts_dict = {'chain': chain}

    elif args.model == "planarRobot4p_10":
        conf_likelihood_var = 4e-2 * np.ones(10)
        conf_likelihood_var[0] = 1
        from scipy.stats import multivariate_normal
        x0 = multivariate_normal(np.zeros(10), conf_likelihood_var * np.eye(10)).rvs(1)
        nuts = sampyl.NUTS(lnpdf, start={"theta": x0})
        chain = []
        n_fevals = []
        start = time()
        timestamps = []
        n_samps_per_iter = 100
        while len(chain) * n_samps_per_iter < args.mcmc_nsamps:
            if args.mcmc_nsamps > len(chain) + n_samps_per_iter:
                chain.append(np.vstack([samp[0] for samp in nuts.sample(n_samps_per_iter, burn=0)]))
            else:
                chain.append(np.vstack([samp[0] for samp in nuts.sample(args.mcmc_nsamps - len(chain) * n_samps_per_iter, burn=0)]))
            timestamps.append(time() - start)
            n_fevals.append(lnpdf.counter)
            np.savez(args.model + args.progressDir + "NUTS_" + str(len(chain)*n_samps_per_iter ) + "of" + str(args.mcmc_nsamps) + ".npz",
                     samples=np.array(chain), fevals=np.array(n_fevals),
                     timestamps=np.array(timestamps))
        np.savez(args.model + args.progressDir + "NUTS_" + str(args.mcmc_nsamps) + "processed_data.npz",
                 samples=np.array(chain),
                 fevals=np.array(n_fevals), timestamps=np.array(timestamps))
        #   lls       = np.array([ lnpdf_grad(*c) for c in chain ])
        nuts_dict = {'chain': chain}
        nuts_dict = {'chain': chain}

    elif args.model == "planarRobot_3":
        conf_likelihood_var = 4e-2 * np.ones(3)
        conf_likelihood_var[0] = 1
        from scipy.stats import multivariate_normal
        x0 = multivariate_normal(np.zeros(3), conf_likelihood_var * np.eye(3)).rvs(1)
        start = time()
        timestamps = []
        n_fevals = []
        chain = np.empty((0,D))
        nuts = sampyl.NUTS(lnpdf, start={"theta": x0})
        chain = []
        n_samps_per_iter = 1000
        while len(chain) * n_samps_per_iter < args.mcmc_nsamps:
            if args.mcmc_nsamps > len(chain) + n_samps_per_iter:
                chain.append(np.vstack([samp[0] for samp in nuts.sample(n_samps_per_iter, burn=0)]))
            else:
                chain.append(np.vstack([samp[0] for samp in nuts.sample(args.mcmc_nsamps - len(chain) * n_samps_per_iter, burn=0)]))
            timestamps.append(time() - start)
            n_fevals.append(lnpdf.counter)
            np.savez(args.model + args.progressDir + "NUTS_" + str(len(chain)*n_samps_per_iter ) + "of" + str(args.mcmc_nsamps) + ".npz",
                     samples=np.array(chain), fevals=np.array(n_fevals),
                     timestamps = np.array(timestamps))
        np.savez(args.model + args.progressDir + "NUTS_" + str(args.mcmc_nsamps) +  "processed_data.npz", samples=np.array(chain),
                 fevals=np.array(n_fevals), timestamps=np.array(timestamps))
        #   lls       = np.array([ lnpdf_grad(*c) for c in chain ])
        nuts_dict = {'chain': chain}
        nuts_dict = {'chain': chain}
    elif args.model == "GMM_2" or args.model == "GMM_20":
        from scipy.stats import multivariate_normal
        x0 = multivariate_normal(np.zeros(D), 1e3 * np.eye(D)).rvs(1)
        start = time()
        timestamps = []
        chain = np.empty((0,D))
        nuts = sampyl.NUTS(lnpdf, start={"theta": x0})
        chain = []
        n_samps_per_iter = 1000
        while len(chain) * n_samps_per_iter < args.mcmc_nsamps:
            if args.mcmc_nsamps > len(chain) + n_samps_per_iter:
                chain.append(np.vstack([samp[0] for samp in nuts.sample(n_samps_per_iter, burn=0)]))
            else:
                chain.append(np.vstack([samp[0] for samp in nuts.sample(args.mcmc_nsamps - len(chain)*n_samps_per_iter, burn=0)]))
            timestamps.append(time() - start)
            np.savez(args.model + args.progressDir + "NUTS_" + str(len(chain)*n_samps_per_iter ) + "of" + str(args.mcmc_nsamps) + ".npz",
                     samples=np.array(chain), fevals=lnpdf.counter,
                     walltime=timestamps)
        np.savez(args.model + args.progressDir + "NUTS_" + str(args.mcmc_nsamps) + ".npz", samples=np.array(chain),
                 fevals=lnpdf.counter, timestamps=timestamps)
        #   lls       = np.array([ lnpdf_grad(*c) for c in chain ])
        nuts_dict = {'chain': chain}
        nuts_dict = {'chain': chain}
    elif args.model == "breast_cancer":
        chain = []
        start = time()
        timestamps = []
        n_fevals = []
        nuts      = sampyl.NUTS(lnpdf, start={"theta": np.random.randn(D)*.1})
        n_samps_per_iter = 100
        while len(chain)*n_samps_per_iter < args.mcmc_nsamps:
            if args.mcmc_nsamps > len(chain) + n_samps_per_iter:
                chain.append(np.vstack([samp[0] for samp in nuts.sample(n_samps_per_iter, burn=0)]))
            else:
                chain.append(np.vstack([samp[0] for samp in nuts.sample(args.mcmc_nsamps - len(chain)*n_samps_per_iter, burn=0)]))
            timestamps.append(time() - start)
            n_fevals.append(lnpdf.counter)
            np.savez(args.model + args.progressDir + "NUTS_" + str(len(chain)*n_samps_per_iter ) + "of" + str(args.mcmc_nsamps) + ".npz",
                     samples=np.array(chain), fevals=np.array(n_fevals),
                     timestamps=np.array(timestamps))
        np.savez(args.model + args.progressDir + "NUTS_" + str(args.mcmc_nsamps) + "processed_data.npz",
                 samples=np.array(chain),
                 fevals=np.array(n_fevals), timestamps=np.array(timestamps))
        nuts_dict = {'chain': chain}
    elif args.model == "iono":
        from experiments.lnpdfs.create_target_lnpfs import build_GPR_iono_with_grad_lnpdf_no_autograd
        lnpdf_tmp = build_GPR_iono_with_grad_lnpdf_no_autograd()
        def lnpdf_grad(theta):
            return lnpdf_tmp(theta)
        chain = []
        start = time()
        timestamps = []
        n_fevals = []
        nuts      = sampyl.NUTS(lnpdf_grad, grad_logp=True, start={"theta": np.random.randn(D)*.1})
        n_samps_per_iter = 100
        while len(chain)*n_samps_per_iter < args.mcmc_nsamps:
            if args.mcmc_nsamps > len(chain) + n_samps_per_iter:
                chain.append(np.vstack([samp[0] for samp in nuts.sample(n_samps_per_iter, burn=0)]))
            else:
                chain.append(np.vstack([samp[0] for samp in nuts.sample(args.mcmc_nsamps - len(chain)*n_samps_per_iter, burn=0)]))
            timestamps.append(time() - start)
            n_fevals.append(lnpdf.counter)
            np.savez(args.model + args.progressDir + "NUTS_" + str(len(chain)*n_samps_per_iter ) + "of" + str(args.mcmc_nsamps) + ".npz",
                     samples=np.array(chain), fevals=np.array(n_fevals),
                     timestamps=np.array(timestamps))
        np.savez(args.model + args.progressDir + "NUTS_" + str(args.mcmc_nsamps) + "processed_data.npz",
                 samples=np.array(chain),
                 fevals=np.array(n_fevals), timestamps=np.array(timestamps))
        nuts_dict = {'chain': chain}

    mcmc_file = os.path.join(args.output, 'mcmc.pkl')
    with open(mcmc_file, 'wb') as f:
        pickle.dump(nuts_dict, f)


