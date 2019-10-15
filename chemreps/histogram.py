'''
Function for creating histogram representations

Literature References:
    - DOI: Need correct one

Disclaimers:
    - This only works for mdl/sdf type files
    - This is an attempt at the recreation from literature and may not be
      implemented as exactly as it is in the literature source
    - This is also not a finished product and is still being worked on


TODO:
1. Extract feature data
2. Make histograms for each feature
3. Find min/max of histograms
4. Make feature vector component from histogram
'''
import copy
import glob
import numpy as np
from scipy.signal import argrelextrema
from .utils.molecule import Molecule
from .utils.calcs import length
from .utils.calcs import angle
from .utils.calcs import torsion
from sklearn.neighbors.kde import KernelDensity
from scipy.stats import norm
from itertools import chain
import time
import pickle


def d_dataset(dataset):
    '''
    Generates all distance data for the entire dataset

    Parameters
    ---------
    dataset: path
        path to all molecules in the dataset

    Returns
    -------
    dist_info: dict
        dict of all two body interactions and list of corresponding length values in dataset
    '''
    # iterate through all of the molecules in the dataset
    #   and get the sizes of the largest bags
    dist_info = {}
    for mol_file in glob.iglob("{}/*".format(dataset)):
        current_molecule = Molecule(mol_file)

        # grab bonds/nonbonds
        for i in range(current_molecule.n_atom):
            for j in range(i, current_molecule.n_atom):
                atomi = current_molecule.sym[i]
                atomj = current_molecule.sym[j]
                if i != j:
                    if atomj < atomi:
                        atomi, atomj = atomj, atomi
                    bond = "{}{}".format(atomi, atomj)
                    rij = length(current_molecule, i, j)
                    if bond not in dist_info:
                        dist_info[bond] = [rij]
                    else:
                        dist_info[bond].append(rij)

    return dist_info


def dat_dataset(dataset):
    '''
    Generates all distance, angle, and torsion data for the entire dataset

    Parameters
    ---------
    dataset: path
        path to all molecules in the dataset

    Returns
    -------
    bat_info: dict
        dict of all bonds, angles, and torsions
    '''
    # iterate through all of the molecules in the dataset
    #   and get the sizes of the largest bags
    dist_info = {}
    angle_info = {}
    torsion_info = {}
    for mol_file in glob.iglob("{}/*".format(dataset)):
        current_molecule = Molecule(mol_file)
        if current_molecule.ftype != 'sdf':
            raise NotImplementedError(
                'file type \'{}\'  is unsupported. Accepted formats: sdf.'.format(current_molecule.ftype))

        # grab bonds/nonbonds
        for i in range(current_molecule.n_atom):
            for j in range(i, current_molecule.n_atom):
                atomi = current_molecule.sym[i]
                atomj = current_molecule.sym[j]
                if i != j:
                    if atomj < atomi:
                        atomi, atomj = atomj, atomi
                    bond = "{}{}".format(atomi, atomj)
                    rij = length(current_molecule, i, j)
                    if bond not in dist_info:
                        dist_info[bond] = [rij]
                    else:
                        dist_info[bond].append(rij)

        # grab angles
        angles = []
        angval = []
        for i in range(current_molecule.n_connect):
            # This is a convoluted way of grabing angles but was one of the
            # fastest. The connectivity is read through and all possible
            # connections are made based on current_molecule.connect.
            # current_molecule.connect then gets translated into
            # current_molecule.sym to make bags based off of atom symbols
            connect = []
            for j in range(current_molecule.n_connect):
                if i in current_molecule.connect[j]:
                    if i == current_molecule.connect[j][0]:
                        connect.append(int(current_molecule.connect[j][1]))
                    elif i == current_molecule.connect[j][1]:
                        connect.append(int(current_molecule.connect[j][0]))
            if len(connect) > 1:
                for k in range(len(connect)):
                    for l in range(k + 1, len(connect)):
                        k_c = connect[k] - 1
                        i_c = i - 1
                        l_c = connect[l] - 1
                        a = current_molecule.sym[k_c]
                        b = current_molecule.sym[i_c]
                        c = current_molecule.sym[l_c]
                        if c < a:
                            # swap for lexographic order
                            a, c = c, a
                        abc = a + b + c
                        ang_theta = angle(current_molecule, k_c, i_c, l_c)
                        angles.append(abc)
                        angval.append(ang_theta)

        for i in range(len(angles)):
            if angles[i] not in angle_info:
                angle_info[angles[i]] = [angval[i]]
            else:
                angle_info[angles[i]].append(angval[i])

        # grab torsions
        # This generates all torsions based on current_molecule.connect
        # not on the current_molecule.sym (atom type)
        tors = []
        for i in range(current_molecule.n_connect):
            # Iterate through the list of connected files and store
            # them as b and c for an abcd torsion
            b = int(current_molecule.connect[i][0])
            c = int(current_molecule.connect[i][1])
            for j in range(current_molecule.n_connect):
                # Join connected values on b of bc to make abc .
                # Below is done twice, swapping which to join on
                # to make sure and get all possibilities
                if int(current_molecule.connect[j][0]) == b:
                    a = int(current_molecule.connect[j][1])
                    # Join connected values on c of abc to make abcd.
                    # Below is done twice, swapping which to join on
                    # to make sure and get all possibilities
                    for k in range(current_molecule.n_connect):
                        if int(current_molecule.connect[k][0]) == c:
                            d = int(current_molecule.connect[k][1])
                            abcd = [a, b, c, d]
                            if len(abcd) == len(set(abcd)):
                                tors.append(abcd)
                    for k in range(current_molecule.n_connect):
                        if int(current_molecule.connect[k][1]) == c:
                            d = int(current_molecule.connect[k][0])
                            abcd = [a, b, c, d]
                            if len(abcd) == len(set(abcd)):
                                tors.append(abcd)
                elif int(current_molecule.connect[j][1]) == b:
                    a = int(current_molecule.connect[j][0])
                    for k in range(current_molecule.n_connect):
                        if int(current_molecule.connect[k][0]) == c:
                            d = int(current_molecule.connect[k][1])
                            abcd = [a, b, c, d]
                            if len(abcd) == len(set(abcd)):
                                tors.append(abcd)
                    for k in range(current_molecule.n_connect):
                        if int(current_molecule.connect[k][1]) == c:
                            d = int(current_molecule.connect[k][0])
                            abcd = [a, b, c, d]
                            if len(abcd) == len(set(abcd)):
                                tors.append(abcd)

        torsions = []
        torval = []
        # This translates all of the torsions from current_molecule.connect
        # to their symbol in order to make bags based upon the symbol
        for i in range(len(tors)):
            a = tors[i][0] - 1
            b = tors[i][1] - 1
            c = tors[i][2] - 1
            d = tors[i][3] - 1
            a_sym = current_molecule.sym[a]
            b_sym = current_molecule.sym[b]
            c_sym = current_molecule.sym[c]
            d_sym = current_molecule.sym[d]
            if d_sym < a_sym:
                # swap for lexographic order
                a_sym, b_sym, c_sym, d_sym = d_sym, c_sym, b_sym, a_sym
            abcd = a_sym + b_sym + c_sym + d_sym
            tor_theta = torsion(current_molecule, a, b, c, d)
            torsions.append(abcd)
            torval.append(tor_theta)
        for i in range(len(torsions)):
            if torsions[i] not in torsion_info:
                torsion_info[torsions[i]] = [torval[i]]
            else:
                torsion_info[torsions[i]].append(torval[i])

    # combine all the information into one dictionary
    bat_info = dist_info.copy()
    bat_info.update(angle_info)
    bat_info.update(torsion_info)

    return bat_info


def dist_feats(mol_file):
    '''
    This returns the two body distance informaiton for the molecule

    Parameters
    ---------
    mol_file: path
        path to molecule

    Returns
    -------
    dist_info: dict
        dict of all two body distances for the molecule
    '''
    # iterate through all of the molecules in the dataset
    #   and get the sizes of the largest bags
    dist_info = {}
    current_molecule = Molecule(mol_file)

    # grab bonds/nonbonds
    for i in range(current_molecule.n_atom):
        for j in range(i, current_molecule.n_atom):
            atomi = current_molecule.sym[i]
            atomj = current_molecule.sym[j]
            if i != j:
                if atomj < atomi:
                    atomi, atomj = atomj, atomi
                bond = "{}{}".format(atomi, atomj)
                rij = length(current_molecule, i, j)
                if bond not in dist_info:
                    dist_info[bond] = [rij]
                else:
                    dist_info[bond].append(rij)
    return dist_info


def dat_feats(mol_file):
    '''
    This gets out the distance, angle, diehdral infomration for the molecule

    Parameters
    ---------
    mol_file: path
        path to molecule

    Returns
    -------
    bat_info: dict
        dict of all bonds, angles, and torsions for the molecule
    '''
    # iterate through all of the molecules in the dataset
    #   and get the sizes of the largest bags
    dist_info = {}
    angle_info = {}
    torsion_info = {}
    current_molecule = Molecule(mol_file)
    if current_molecule.ftype != 'sdf':
        raise NotImplementedError(
            'file type \'{}\'  is unsupported. Accepted formats: sdf.'.format(current_molecule.ftype))

    # grab bonds/nonbonds
    for i in range(current_molecule.n_atom):
        for j in range(i, current_molecule.n_atom):
            atomi = current_molecule.sym[i]
            atomj = current_molecule.sym[j]
            if i != j:
                if atomj < atomi:
                    atomi, atomj = atomj, atomi
                bond = "{}{}".format(atomi, atomj)
                rij = length(current_molecule, i, j)
                if bond not in dist_info:
                    dist_info[bond] = [rij]
                else:
                    dist_info[bond].append(rij)

    # grab angles
    angles = []
    angval = []
    for i in range(current_molecule.n_connect):
        # This is a convoluted way of grabing angles but was one of the
        # fastest. The connectivity is read through and all possible
        # connections are made based on current_molecule.connect.
        # current_molecule.connect then gets translated into
        # current_molecule.sym to make bags based off of atom symbols
        connect = []
        for j in range(current_molecule.n_connect):
            if i in current_molecule.connect[j]:
                if i == current_molecule.connect[j][0]:
                    connect.append(int(current_molecule.connect[j][1]))
                elif i == current_molecule.connect[j][1]:
                    connect.append(int(current_molecule.connect[j][0]))
        if len(connect) > 1:
            for k in range(len(connect)):
                for l in range(k + 1, len(connect)):
                    k_c = connect[k] - 1
                    i_c = i - 1
                    l_c = connect[l] - 1
                    a = current_molecule.sym[k_c]
                    b = current_molecule.sym[i_c]
                    c = current_molecule.sym[l_c]
                    if c < a:
                        # swap for lexographic order
                        a, c = c, a
                    abc = a + b + c
                    ang_theta = angle(current_molecule, k_c, i_c, l_c)
                    angles.append(abc)
                    angval.append(ang_theta)

    for i in range(len(angles)):
        if angles[i] not in angle_info:
            angle_info[angles[i]] = [angval[i]]
        else:
            angle_info[angles[i]].append(angval[i])

    # grab torsions
    # This generates all torsions based on current_molecule.connect
    # not on the current_molecule.sym (atom type)
    tors = []
    for i in range(current_molecule.n_connect):
        # Iterate through the list of connected files and store
        # them as b and c for an abcd torsion
        b = int(current_molecule.connect[i][0])
        c = int(current_molecule.connect[i][1])
        for j in range(current_molecule.n_connect):
            # Join connected values on b of bc to make abc .
            # Below is done twice, swapping which to join on
            # to make sure and get all possibilities
            if int(current_molecule.connect[j][0]) == b:
                a = int(current_molecule.connect[j][1])
                # Join connected values on c of abc to make abcd.
                # Below is done twice, swapping which to join on
                # to make sure and get all possibilities
                for k in range(current_molecule.n_connect):
                    if int(current_molecule.connect[k][0]) == c:
                        d = int(current_molecule.connect[k][1])
                        abcd = [a, b, c, d]
                        if len(abcd) == len(set(abcd)):
                            tors.append(abcd)
                for k in range(current_molecule.n_connect):
                    if int(current_molecule.connect[k][1]) == c:
                        d = int(current_molecule.connect[k][0])
                        abcd = [a, b, c, d]
                        if len(abcd) == len(set(abcd)):
                            tors.append(abcd)
            elif int(current_molecule.connect[j][1]) == b:
                a = int(current_molecule.connect[j][0])
                for k in range(current_molecule.n_connect):
                    if int(current_molecule.connect[k][0]) == c:
                        d = int(current_molecule.connect[k][1])
                        abcd = [a, b, c, d]
                        if len(abcd) == len(set(abcd)):
                            tors.append(abcd)
                for k in range(current_molecule.n_connect):
                    if int(current_molecule.connect[k][1]) == c:
                        d = int(current_molecule.connect[k][0])
                        abcd = [a, b, c, d]
                        if len(abcd) == len(set(abcd)):
                            tors.append(abcd)

    torsions = []
    torval = []
    # This translates all of the torsions from current_molecule.connect
    # to their symbol in order to make bags based upon the symbol
    for i in range(len(tors)):
        a = tors[i][0] - 1
        b = tors[i][1] - 1
        c = tors[i][2] - 1
        d = tors[i][3] - 1
        a_sym = current_molecule.sym[a]
        b_sym = current_molecule.sym[b]
        c_sym = current_molecule.sym[c]
        d_sym = current_molecule.sym[d]
        if d_sym < a_sym:
            # swap for lexographic order
            a_sym, b_sym, c_sym, d_sym = d_sym, c_sym, b_sym, a_sym
        abcd = a_sym + b_sym + c_sym + d_sym
        tor_theta = torsion(current_molecule, a, b, c, d)
        torsions.append(abcd)
        torval.append(tor_theta)
    for i in range(len(torsions)):
        if torsions[i] not in torsion_info:
            torsion_info[torsions[i]] = [torval[i]]
        else:
            torsion_info[torsions[i]].append(torval[i])

    # combine all the information into one dictionary
    bat_info = dist_info.copy()
    bat_info.update(angle_info)
    bat_info.update(torsion_info)

    return bat_info


def bin_nphisto(data):
    '''
    Using Numpy's histogram function for bin generation.
    This tends to yield a lot of bins making the 
    ..representation extremely large.

    Parameters
    ---------
    data: list
        all information for that feature

    Returns
    -------
    edges: list
        list of all bin edges
    empty: list
        list of empty bins for filling
    '''
    a = np.histogram(data, bins='auto')

    lmax = list(argrelextrema(a[0], np.greater)[0])
    lmin = list(argrelextrema(a[0], np.less)[0])
    lmax.extend(lmin)
    idx = sorted(lmax)

    minmax = np.array([a[1][idx[i]] for i in range(len(idx))])
    minmax = np.insert(minmax, 0, 0)
    minmax = np.append(minmax, data.max())
    edges = (minmax[1:] + minmax[:-1]) / 2
    edges = np.insert(edges, 0, 0)
    edges = np.append(edges, data.max() + .1)
    empty = np.zeros(len(edges) + 1)

    return list(edges), list(empty)


def bin_kde(data, kernel='gaussian', bandwidth=.1):
    '''
    Kernel Density Estimation bin generation.
    The kernel and bandwidth can be used to fine tune bins.
    This method does take a large amount of time.

    Parameters
    ---------
    data: list
        all information for that feature
    kernel: str
        kernel parameter for bin generation using kernel density estimation
    bandwidth: float
        bandwidth parameter for bin generation using kernel density estimation

    Returns
    -------
    edges: list
        list of all bin edges
    empty: list
        list of empty bins for filling
    '''
    data = np.array(data)[:, np.newaxis]
    b = np.linspace(data.min(), data.max(), 1000)[:, np.newaxis]

    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(data)
    log_dens = kde.score_samples(b)

    lmax = argrelextrema(np.exp(log_dens), np.greater)[0]
    lmin = argrelextrema(np.exp(log_dens), np.less)[0]
    lmax.extend(lmin)
    idx = sorted(lmax)

    minmax = np.array([b[idx[i]] for i in range(len(idx))])
    minmax = np.insert(minmax, 0, 0)
    minmax = np.append(minmax, data.max())
    edges = (minmax[1:] + minmax[:-1]) / 2
    edges = np.insert(edges, 0, 0)
    edges = np.append(edges, data.max() + .1)
    empty = np.zeros(len(edges) + 1)

    return list(edges), list(empty)


def bin_generator(bin_type, dataset, binmeth='kde', kernel='gaussian', bandwidth=.1):
    '''
    Bin generator for histogram methods 

    Parameters
    ---------
    bin_type: str
        what kind of binds you want for your data ('D' or 'DAT')
    dataset: str
        path to all molecules in the dataset
    binmeth: str
        determines bin parameter method
    kernel: str
        kernel parameter for bin generation using kernel density estimation
    bandwidth: float
        bandwidth parameter for bin generation using kernel density estimation

    Returns
    -------
    bins: dict
        dictionary of bin edges
    empty_bins: dict
        dictionary of empty bins for filling
    '''
    if bin_type == 'D':
        dataset_info = d_dataset(dataset)
    elif bin_type == 'DAT':
        dataset_info = dat_dataset(dataset)
    else:
        raise NotImplementedError(
            'bin type \'{}\'  is unsupported. Accepted bins: D and dat.'.format(bin_type))

    edge_list = []
    empty_list = []
    for k in dataset_info.keys():
        if binmeth == 'kde':
            edges, empty = bin_kde(
                dataset_info[k], kernel=kernel, bandwidth=bandwidth)
        elif binmeth == 'nphisto':
            edges, empty = bin_nphisto(dataset_info[k])
        edge_list.append([k, edges])
        empty_list.append([k, empty])

    bins = {t[0]: t[1:] for t in edge_list}
    empty_bins = {t[0]: t[1:] for t in empty_list}
    return bins, empty_bins


def hd(mol, bin_type, bins, empty_bins):
    '''
    Fill bins with information from mol_feats
    '''
    if bin_type == 'D':
        data = dist_feats(mol)
    elif bin_type == 'DAT':
        data = dat_feats(mol)
    else:
        raise NotImplementedError(
            'bin type \'{}\'  is unsupported. Accepted bins: D and DAT.'.format(bin_type))

    filling = copy.deepcopy(empty_bins)
    for k in data.keys():
        for i in range(len(bins[k][0])):
            for j in range(len(data[k])):
                if data[k][j] > bins[k][0][i] and data[k][j] < bins[k][0][i+1]:
                    if i < len(bins[k][0]) - 1:
                        dist = bins[k][0][i+1] - bins[k][0][i]
                        val1 = (bins[k][0][i+1] - data[k][j]) / dist
                        val2 = 1 - val1
                        filling[k][0][i] += val1
                        filling[k][0][i+1] += val2
        # TODO
        # There seems to be issues with this
        # Only filling about half the bags it is supposed to
        # Need to sort mol_features
        # mol_data = sorted(data[k])
        # # # print(mol_data)
        # j = 0  # used for indexing bin_edges
        # for i in range(len(mol_data)):
        #     if mol_data[i] <= bins[k][0][j + 1]:
        #         dist = bins[k][0][j + 1] - bins[k][0][j]
        #         val1 = (bins[k][0][j + 1] - mol_data[i]) / dist
        #         val2 = 1 - val1
        #         filling[k][0][j] += val1
        #         filling[k][0][j + 1] += val2
        #     elif j > len(bins[k][0]):
        #         val = mol_data[i] - bins[k][0][j]
        #         filling[k][0][j] += val
        #     while mol_data[i] > bins[k][0][j + 1]:
        #         # j += 1
        #         if mol_data[i] > bins[k][0][-1]:
        #             # if j > len(bin_edges[k][0]):
        #             val = mol_data[i] - bins[k][0][j]
        #             filling[k][0][-1] += val
        #             break
        #         elif mol_data[i] < bins[k][0][j + 1]:
        #             dist = bins[k][0][j + 1] - bins[k][0][j]
        #             val1 = (bins[k][0][j + 1] - mol_data[i]) / dist
        #             val2 = 1 - val1
        #             filling[k][0][j] += val1
        #             filling[k][0][j + 1] += val2
        #         j += 1
        #         # elif j > len(bin_edges[k][0]):
        #         #     val = mol_data[i] - bin_edges[k][0][j]
        #         #     filling[k][0][j] += val

    full_bin = []
    bin_keys = list(filling.keys())
    for i in range(len(bin_keys)):
        full_bin.append(filling[bin_keys[i]][0])

    full_bin = np.array(list(chain.from_iterable(full_bin)), dtype=np.float16)
    return full_bin
