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
# import jenkspy
# from jenks import jenks
# import random


def hist_maker(dataset):
    '''
    Parameters
    ---------
    dataset: path
        path to all molecules in the dataset

    Returns
    -------
    bond_info: dict
        dict of all bonds and list of corresponding length values in dataset
    angle_info: dict
        dict of all angles and list of corresponding angle values in dataset
    torsion_info: dict
        dict of all torsions and list of corresponding torsion values in dataset
    '''
    # iterate through all of the molecules in the dataset
    #   and get the sizes of the largest bags
    bond_info = {}
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
                zi = current_molecule.at_num[i]
                zj = current_molecule.at_num[j]
                if i != j:
                    if atomj < atomi:
                        atomi, atomj = atomj, atomi
                    bond = "{}{}".format(atomi, atomj)
                    rij = length(current_molecule, i, j)
                    if bond not in bond_info:
                        bond_info[bond] = [rij]
                    else:
                        bond_info[bond].append(rij)
        ###### Uncommment once binning method is working for bonds ######
        # # grab angles
        # angles = []
        # angval = []
        # for i in range(current_molecule.n_connect):
        #     # This is a convoluted way of grabing angles but was one of the
        #     # fastest. The connectivity is read through and all possible
        #     # connections are made based on current_molecule.connect.
        #     # current_molecule.connect then gets translated into
        #     # current_molecule.sym to make bags based off of atom symbols
        #     connect = []
        #     for j in range(current_molecule.n_connect):
        #         if i in current_molecule.connect[j]:
        #             if i == current_molecule.connect[j][0]:
        #                 connect.append(int(current_molecule.connect[j][1]))
        #             elif i == current_molecule.connect[j][1]:
        #                 connect.append(int(current_molecule.connect[j][0]))
        #     if len(connect) > 1:
        #         for k in range(len(connect)):
        #             for l in range(k + 1, len(connect)):
        #                 k_c = connect[k] - 1
        #                 i_c = i - 1
        #                 l_c = connect[l] - 1
        #                 a = current_molecule.sym[k_c]
        #                 b = current_molecule.sym[i_c]
        #                 c = current_molecule.sym[l_c]
        #                 if c < a:
        #                     # swap for lexographic order
        #                     a, c = c, a
        #                 abc = a + b + c
        #                 ang_theta = angle(current_molecule, k_c, i_c, l_c)
        #                 angles.append(abc)
        #                 angval.append(ang_theta)
        #
        # for i in range(len(angles)):
        #     if angles[i] not in angle_info:
        #         angle_info[angles[i]] = [angval[i]]
        #     else:
        #         angle_info[angles[i]].append(angval[i])
        #
        # # grab torsions
        # # This generates all torsions based on current_molecule.connect
        # # not on the current_molecule.sym (atom type)
        # tors = []
        # for i in range(current_molecule.n_connect):
        #     # Iterate through the list of connected files and store
        #     # them as b and c for an abcd torsion
        #     b = int(current_molecule.connect[i][0])
        #     c = int(current_molecule.connect[i][1])
        #     for j in range(current_molecule.n_connect):
        #         # Join connected values on b of bc to make abc .
        #         # Below is done twice, swapping which to join on
        #         # to make sure and get all possibilities
        #         if int(current_molecule.connect[j][0]) == b:
        #             a = int(current_molecule.connect[j][1])
        #             # Join connected values on c of abc to make abcd.
        #             # Below is done twice, swapping which to join on
        #             # to make sure and get all possibilities
        #             for k in range(current_molecule.n_connect):
        #                 if int(current_molecule.connect[k][0]) == c:
        #                     d = int(current_molecule.connect[k][1])
        #                     abcd = [a, b, c, d]
        #                     if len(abcd) == len(set(abcd)):
        #                         tors.append(abcd)
        #             for k in range(current_molecule.n_connect):
        #                 if int(current_molecule.connect[k][1]) == c:
        #                     d = int(current_molecule.connect[k][0])
        #                     abcd = [a, b, c, d]
        #                     if len(abcd) == len(set(abcd)):
        #                         tors.append(abcd)
        #         elif int(current_molecule.connect[j][1]) == b:
        #             a = int(current_molecule.connect[j][0])
        #             for k in range(current_molecule.n_connect):
        #                 if int(current_molecule.connect[k][0]) == c:
        #                     d = int(current_molecule.connect[k][1])
        #                     abcd = [a, b, c, d]
        #                     if len(abcd) == len(set(abcd)):
        #                         tors.append(abcd)
        #             for k in range(current_molecule.n_connect):
        #                 if int(current_molecule.connect[k][1]) == c:
        #                     d = int(current_molecule.connect[k][0])
        #                     abcd = [a, b, c, d]
        #                     if len(abcd) == len(set(abcd)):
        #                         tors.append(abcd)
        #
        # torsions = []
        # torval = []
        # # This translates all of the torsions from current_molecule.connect
        # # to their symbol in order to make bags based upon the symbol
        # for i in range(len(tors)):
        #     a = tors[i][0] - 1
        #     b = tors[i][1] - 1
        #     c = tors[i][2] - 1
        #     d = tors[i][3] - 1
        #     a_sym = current_molecule.sym[a]
        #     b_sym = current_molecule.sym[b]
        #     c_sym = current_molecule.sym[c]
        #     d_sym = current_molecule.sym[d]
        #     if d_sym < a_sym:
        #         # swap for lexographic order
        #         a_sym, b_sym, c_sym, d_sym = d_sym, c_sym, b_sym, a_sym
        #     abcd = a_sym + b_sym + c_sym + d_sym
        #     tor_theta = torsion(current_molecule, a, b, c, d)
        #     # print(a, b, c, d, a_sym, b_sym, c_sym, d_sym, tor_theta)
        #     torsions.append(abcd)
        #     torval.append(tor_theta)
        # for i in range(len(torsions)):
        #     if torsions[i] not in torsion_info:
        #         torsion_info[torsions[i]] = [torval[i]]
        #     else:
        #         torsion_info[torsions[i]].append(torval[i])

    return bond_info#, angle_info, torsion_info

def mol_feat(mol_file):
    '''
    Parameters
    ---------
    dataset: path
        path to all molecules in the dataset

    Returns
    -------
    bond_info: dict
        dict of all bonds and list of corresponding length values in dataset
    angle_info: dict
        dict of all angles and list of corresponding angle values in dataset
    torsion_info: dict
        dict of all torsions and list of corresponding torsion values in dataset
    '''
    # iterate through all of the molecules in the dataset
    #   and get the sizes of the largest bags
    bond_info = {}
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
            zi = current_molecule.at_num[i]
            zj = current_molecule.at_num[j]
            if i != j:
                if atomj < atomi:
                    atomi, atomj = atomj, atomi
                bond = "{}{}".format(atomi, atomj)
                rij = length(current_molecule, i, j)
                if bond not in bond_info:
                    bond_info[bond] = [rij]
                else:
                    bond_info[bond].append(rij)
    return bond_info


def bin_nphisto(data):
    '''
    Attempt using Numpy's histogram in order to generate bins
    '''
    a = np.histogram(data, bins='auto')
    # print(a)
    # lmax = argrelextrema(a[0], np.greater)[0]
    # lmin = argrelextrema(a[0], np.less)[0]

    # idx = []
    # for i in range(len(lmin)):
    #     idx.append(lmin[i])
    # for j in range(len(lmax)):
    #     idx.append(lmax[j])
    # idx = sorted(idx)
    #
    # bin_minmax = np.array([a[1][idx[i]] for i in range(len(idx))])
    # bin_minmax = np.insert(bin_minmax, 0, 0)
    # bin_minmax = np.append(bin_minmax, data.max())
    # bin_edges = (bin_minmax[1:] + bin_minmax[:-1]) / 2
    # bin_edges = np.insert(bin_edges, 0, 0)
    # bin_edges = np.append(bin_edges, data.max() + .1)

    lmax = list(argrelextrema(a[0], np.greater)[0])
    lmin = list(argrelextrema(a[0], np.less)[0])
    lmax.extend(lmin)
    idx = sorted(lmax)
    # bin_minmax = np.array([a[1][idx[i]] for i in range(len(idx))])
    # bin_edges = (bin_minmax[1:] + bin_minmax[:-1]) / 2

    bin_minmax = np.array([a[1][idx[i]] for i in range(len(idx))])
    bin_minmax = np.insert(bin_minmax, 0, 0)
    bin_minmax = np.append(bin_minmax, data.max())
    bin_edges = (bin_minmax[1:] + bin_minmax[:-1]) / 2
    bin_edges = np.insert(bin_edges, 0, 0)
    bin_edges = np.append(bin_edges, data.max() + .1)

    # bin_edges = [j for i in bin_edges for j in i]
    empty_bin = np.zeros(len(bin_edges) + 1)

    return list(bin_edges), list(empty_bin)

def bin_kde(data, kernel='gaussian', bandwidth=.1):
    '''
    Attempt using scikit's kernel density estimation
    '''
    # kernel='tophat', bandwidth=1 gives 10 bins
    # kernel='gaussian', bandwidth=.1 gives 27 bins
    # kernel='gaussian', bandwidth=.05 gives 53 bins
    # kernel='gaussian', bandwidth=.07 gives 39 bins
    # kernel='gaussian', bandwidth=.06 gives 43 bins
    # kernel='tophat', bandwidth=.1 gives 239
    # kernel='tophat', bandwidth=.8 gives 23
    # kernel='tophat', bandwidth=.6 gives 25
    # kernel='tophat', bandwidth=.4 gives 83
    # kernel='tophat', bandwidth=.55 gives 42
    data = np.array(data)[:, np.newaxis]
    b = np.linspace(data.min(), data.max(), 1000)[:, np.newaxis]

    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(data)
    log_dens = kde.score_samples(b)

    lmax = argrelextrema(np.exp(log_dens), np.greater)[0]
    lmin = argrelextrema(np.exp(log_dens), np.less)[0]

    idx = []
    for i in range(len(lmin)):
        idx.append(lmin[i])
    for j in range(len(lmax)):
        idx.append(lmax[j])
    idx = sorted(idx)

    bin_minmax = np.array([b[idx[i]] for i in range(len(idx))])
    bin_minmax = np.insert(bin_minmax, 0, 0)
    bin_minmax = np.append(bin_minmax, data.max())
    bin_edges = (bin_minmax[1:] + bin_minmax[:-1]) / 2
    bin_edges = np.insert(bin_edges, 0, 0)
    bin_edges = np.append(bin_edges, data.max() + .1)

    # bin_edges = [j for i in bin_edges for j in i]
    empty_bin = np.zeros(len(bin_edges) + 1)

    return list(bin_edges), list(empty_bin)


def bin_npdict(info):
    a = []
    a_e = []
    for k in info.keys():
        data = np.array(info[k])
        bin_edges, empty_bin = bin_nphisto(data)
        a.append([k, bin_edges])
        a_e.append([k, empty_bin])

    c = {t[0]:t[1:] for t in a}
    c_e = {t[0]:t[1:] for t in a_e}
    return c, c_e

def bin_kdedict(info):
    a = []
    a_e = []
    for k in info.keys():
        data = np.array(info[k])
        bin_edges, empty_bin = bin_kde(data)
        a.append([k, bin_edges])
        a_e.append([k, empty_bin])

    c = {t[0]:t[1:] for t in a}
    c_e = {t[0]:t[1:] for t in a_e}
    return c, c_e


def hd(mol, bin_edges, empty_bin):
    '''
    Fill bins with information from mol_feats
    '''
    data = mol_feat(mol)
    filling = copy.deepcopy(empty_bin)
    for k in data.keys():
        for i in range(len(bin_edges[k][0])):
            for j in range(len(data[k])):
                if data[k][j] > bin_edges[k][0][i] and data[k][j] < bin_edges[k][0][i+1]:
                    if i < len(bin_edges[k][0]) - 1:
                        dist = bin_edges[k][0][i+1] - bin_edges[k][0][i]
                        val1 = (bin_edges[k][0][i+1] - data[k][j]) / dist
                        val2 = 1 - val1
                        filling[k][0][i] += val1
                        filling[k][0][i+1] += val2

    full_bin = []
    bin_keys = list(filling.keys())
    for i in range(len(bin_keys)):
        full_bin.append(filling[bin_keys[i]][0])

    full_bin = np.array(list(chain.from_iterable(full_bin)), dtype=np.float16)
    return full_bin
