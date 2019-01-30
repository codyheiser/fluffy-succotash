# fluffy-succotash utility functions

# @author: C Heiser
# January 2019

# utility functions
from fcc_utils import *


def compare_barcode_distances(obj, barcodes, plot_out=True):
    '''
    Compare Euclidean distance distribution of barcode populations to null population
        obj: object to use for distance calcs. RNA_counts or DR.
        barcodes: which barcodes to compare to null population.
        plot_out: return plot of cumulative distance distributions?
    '''
    mat = obj.distance_matrix(ranks='all')
    flat = mat[np.triu_indices(mat.shape[1],1)] # take upper triangle of matrix and flatten for unique distances
    flat_norm = (flat/flat.max()) # normalize unique distances within set
    out = [] # initate list to make df of EMD and KLD from

    if plot_out: # initiate plot and draw null distribution
        plt.figure(figsize=(5,5))
        # calculate and plot the cumulative probability distributions for cell-cell distances in each dataset
        num_bins = int(len(flat_norm)/100)
        pre_counts, pre_bin_edges = np.histogram(flat_norm, bins=num_bins)
        pre_cdf = np.cumsum(pre_counts)
        plt.plot(pre_bin_edges[1:], pre_cdf/pre_cdf[-1], label='Null Population', linestyle='dashed', linewidth=3, color='black')

    for code in barcodes:
        post = obj.distance_matrix(ranks=[code]) # test population is distance matrix for each code
        # take the upper triangle (it's symmetrical) for calculating EMD and plotting distance differences
        post_flat = post[np.triu_indices(post.shape[1],1)]
        # normalize flattened distances within each set for fair comparison of probability distributions
        post_flat_norm = (post_flat/post_flat.max())
        # calculate EMD for the distance matrices
        EMD = sc.stats.wasserstein_distance(flat_norm, post_flat_norm)
        # Kullback Leibler divergence
        # add very small number to avoid dividing by zero
        KLD = sc.stats.entropy(flat_norm+0.00000001) - sc.stats.entropy(post_flat_norm+0.00000001)
        # append to output frame
        out.append({'code':code, 'EMD':EMD, 'KLD':KLD})

        if plot_out: # draw cumulative distance distribution
            post_counts, post_bin_edges = np.histogram(post_flat_norm, bins=num_bins)
            post_cdf = np.cumsum(post_counts)
            plt.plot(post_bin_edges[1:], post_cdf/post_cdf[-1], label=None, alpha=0.7)

    if plot_out: # add details to plot
        plt.tick_params(labelsize=12)
        plt.legend(loc='best')
        sns.despine()
        plt.tight_layout()
        plt.show()
        plt.close()

    return pd.DataFrame(out)


def cell_bias_curve(obj, pop_sizes, iter):
    '''
    Create standard curve of EMD from random cell distances at different n_cells (pop_sizes) to null population.
        obj: DR object to use for distance calcs
        pop_sizes: list of n_cells values to build curve on (e.g. [10,20,30,40,50] or np.arange(10,51,10))
        iter: number of iterations to perform at each pop_size. used to accurately estimate variability.
    '''
    mat = obj.distance_matrix(ranks='all') # null is whole untreated population
    flat = mat[np.triu_indices(mat.shape[1],1)]
    flat_norm = (flat/flat.max())

    out = [] # initiate output list
    for rep in np.arange(1,iter+1): # number of iterations defined by user
        EMD = [] # initate list to make df of EMDs

        for test_i in pop_sizes: # n_cells values defined by user
            rand = obj.results[np.random.choice(obj.results.shape[0], test_i, replace=False),]
            post = sc.spatial.distance_matrix(rand, rand) # calculate Euclidean distances
            # take the upper triangle (it's symmetrical) for calculating EMD and plotting distance differences
            post_flat = post[np.triu_indices(post.shape[1],1)]
            # normalize flattened distances within each set for fair comparison of probability distributions
            post_flat_norm = (post_flat/post_flat.max())
            # calculate EMD for the distance matrices
            EMD_i = sc.stats.wasserstein_distance(flat_norm, post_flat_norm)
            EMD.append(EMD_i)

        if len(out)==0:
            out = pd.DataFrame({'n_cells':pop_sizes, 'EMD1':EMD})

        else:
            out['EMD{}'.format(rep)] = EMD

    return out
