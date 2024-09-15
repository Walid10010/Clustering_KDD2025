from CLUMPED_Algo import CLUMPED_Start, compare_mod
from sklearn.metrics import adjusted_mutual_info_score,normalized_mutual_info_score
import  numpy as np

from sklearn.preprocessing import MinMaxScaler


def preporcess_Data(X, y):
    dic_ = {}
    X = MinMaxScaler().fit_transform(X)
    for i, item in enumerate(X):
        dic_[tuple(item)] = y[i]

    from functools import cmp_to_key
    X = X.tolist()
    X.sort(key=cmp_to_key(compare_mod))
    y = []
    for item in X:
        y.append(dic_[tuple(item)])
    y = np.array(y).reshape(-1)
    X = np.array(X)
    return X, y

import glob
#First download the data from https://www.dropbox.com/scl/fo/tvsf83a5cusaolxy4pfdu/AGZ-VtU34hsFztS55ZcH2Bw?rlkey=caphnw9083trv54lx9ftbz0i5&st=dj9gduxy&dl=0
#save the data into real-world folder
for data_name in glob.glob('real-world/label*'):
    print(data_name)
    name   = data_name.split('/')[1].split('_')[1]
    print(name)
    X, Y = np.loadtxt('real-world/data_{}.npz'.format(name)), np.loadtxt('real-world/label_{}.npz'.format(name))

    X, Y= preporcess_Data(X, Y)
    y_pred = CLUMPED_Start(X)  # r
    print('NMI:', normalized_mutual_info_score(Y, y_pred))



