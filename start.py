from CBLHALGO import CBLH_Start, compare_mod
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


X, Y = np.loadtxt('data/Intro/SyntheticIntroData'), np.loadtxt('data/Intro/SyntheticIntroLabel')
X, Y= preporcess_Data(X, Y)


y_pred = CBLH_Start(X)  # r
print('AMI:', normalized_mutual_info_score(Y, y_pred))