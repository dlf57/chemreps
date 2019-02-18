from chemreps.bagger import BagMaker
from chemreps.bat import bat
import numpy as np


def test_bag_maker():
    bags_true = {'C': 7, 'CC': 21, 'CCC': 8, 'CCCC': 8, 'CCCH': 12, 'CCCO': 4, 'CCH': 14, 'CCO': 2, 'CCOH': 1, 'CH': 42, 'COH': 1, 'H': 10, 'HCCH': 16, 'HCH': 8, 'HH': 45, 'HOCO': 1, 'HOH': 1, 'O': 2, 'OC': 14, 'OCO': 1, 'OH': 12, 'OO': 1}
    bagger = BagMaker('BAT', 'data/sdf/')
    assert bagger.bag_sizes == bags_true


def test_bat():
    bags_true = {'C': 7, 'CC': 21, 'CCC': 8, 'CCCC': 8, 'CCCH': 12, 'CCCO': 4,
                 'CCH': 14, 'CCO': 2, 'CCOH': 1, 'CH': 42, 'COH': 1, 'H': 10,
                 'HCCH': 16, 'HCH': 8, 'HH': 45, 'HOCO': 1, 'HOH': 1, 'O': 2,
                 'OC': 14, 'OCO': 1, 'OH': 12, 'OO': 1}
    bat_true = np.array([36.84  , 36.84  , 36.84  , 36.84  ,  0.    ,  0.    ,  0.    ,
        0.    , 23.38  , 23.38  , 23.33  , 14.15  , 14.15  ,  9.195 ,
        0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
        0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
        0.    ,  0.    ,  1.943 ,  1.943 ,  0.    ,  0.    ,  0.    ,
        0.    ,  0.    ,  0.    ,  0.    ,  1.    ,  0.    ,  0.    ,
        0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  1.    ,
        1.    ,  0.515 ,  0.515 ,  0.515 ,  0.515 ,  0.4968,  0.4968,
        0.4968,  0.4966,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
        0.    ,  0.    ,  0.    ,  1.932 ,  1.932 ,  1.932 ,  1.932 ,
        1.924 ,  1.924 ,  1.915 ,  1.915 ,  1.915 ,  1.915 ,  1.906 ,
        1.906 ,  1.906 ,  1.906 ,  0.    ,  0.    ,  0.    ,  0.    ,
        0.    ,  0.    ,  5.492 ,  5.492 ,  5.492 ,  5.492 ,  5.492 ,
        5.492 ,  5.492 ,  5.492 ,  5.492 ,  5.492 ,  2.775 ,  2.775 ,
        2.775 ,  2.775 ,  2.762 ,  2.762 ,  2.762 ,  2.762 ,  2.76  ,
        2.76  ,  2.752 ,  2.752 ,  2.752 ,  2.752 ,  2.162 ,  2.162 ,
        2.162 ,  2.162 ,  2.145 ,  2.145 ,  2.145 ,  2.145 ,  1.717 ,
        1.717 ,  1.42  ,  1.42  ,  1.42  ,  1.42  ,  1.272 ,  1.272 ,
        0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.5   ,  0.5   ,
        0.5   ,  0.5   ,  0.5   ,  0.5   ,  0.5   ,  0.5   ,  0.5   ,
        0.5   ,  0.    ,  1.    ,  1.    ,  0.9995,  0.9995,  0.9995,
        0.9995,  0.52  ,  0.5195,  0.5195,  0.5195,  0.4834,  0.4834,
        0.4834,  0.4832,  0.469 ,  0.469 ,  0.    ,  1.8955,  1.8955,
        1.891 ,  1.891 ,  1.891 ,  1.89  ,  1.877 ,  1.877 ,  0.    ,
        0.567 ,  0.567 ,  0.565 ,  0.565 ,  0.565 ,  0.565 ,  0.5635,
        0.5635,  0.4016,  0.4016,  0.4016,  0.4016,  0.3982,  0.3982,
        0.3982,  0.3982,  0.3975,  0.3975,  0.387 ,  0.387 ,  0.387 ,
        0.387 ,  0.3254,  0.3254,  0.3254,  0.3254,  0.3254,  0.3254,
        0.3193,  0.3193,  0.3193,  0.3193,  0.265 ,  0.265 ,  0.265 ,
        0.2646,  0.2256,  0.2256,  0.2095,  0.2095,  0.2041,  0.2041,
        0.2041,  0.2041,  0.1781,  0.    ,  0.    ,  0.    ,  0.    ,
        0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
        0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
        0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
        0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
        0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
        0.    ], dtype=np.float16)

    bagger = BagMaker('BAT', 'data/sdf/')
    assert bagger.bag_sizes == bags_true

    rep = bat('data/sdf/butane.sdf', bagger.bags, bagger.bag_sizes)
    assert np.all(np.abs(bat_true-rep) <= 1e-4) == True


if __name__ == "__main__":
    print("This is a test of the bat representation in chemreps to be evaluated with pytest")