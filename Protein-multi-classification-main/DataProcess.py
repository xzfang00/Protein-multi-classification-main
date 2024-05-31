import scipy
import numpy as np
from collections import Counter
import torch
from sklearn.cluster import MiniBatchKMeans
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from sklearn.decomposition import PCA


def load_data(train_path=None, test_path=None):
    print("[Data Loading]")
    name = "PSTAAP"
    X_train, X_test, Y_train, Y_test = None, None, None, None

    if train_path:
        data = scipy.io.loadmat(train_path)
        d_1 = np.array(data[f'{name}1'])
        d_2 = np.array(data[f'{name}2'])
        d_3 = np.array(data[f'{name}3'])
        d_4 = np.array(data[f'{name}4'])
        d_5 = np.array(data[f'{name}5'])
        d_6 = np.array(data[f'{name}6'])
        d_7 = np.array(data[f'{name}7'])
        d_8 = np.array(data[f'{name}8'])
        d_9 = np.array(data[f'{name}9'])
        d_10 = np.array(data[f'{name}10'])
        d_11 = np.array(data[f'{name}11'])
        X_train = np.concatenate((d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9, d_10, d_11), axis=0)

    if test_path:
        data_test = scipy.io.loadmat(test_path)
        dt_1 = np.array(data_test[f'{name}1'])
        dt_2 = np.array(data_test[f'{name}2'])
        dt_3 = np.array(data_test[f'{name}3'])
        dt_4 = np.array(data_test[f'{name}4'])
        dt_5 = np.array(data_test[f'{name}5'])
        dt_6 = np.array(data_test[f'{name}6'])
        dt_7 = np.array(data_test[f'{name}7'])
        dt_8 = np.array(data_test[f'{name}8'])
        dt_9 = np.array(data_test[f'{name}9'])
        dt_10 = np.array(data_test[f'{name}10'])
        dt_11 = np.array(data_test[f'{name}11'])
        X_test = np.concatenate((dt_1, dt_2, dt_3, dt_4, dt_5, dt_6, dt_7, dt_8, dt_9, dt_10, dt_11), axis=0)

    y_dir = []
    yt_dir = []

    if train_path and test_path:
        for i in range(1, 12):
            exec(f'y_dir.append(np.full(d_{i}.shape[0],i))')

            exec(f'yt_dir.append(np.full(dt_{i}.shape[0],i))')
        Y_train = np.concatenate((y_dir), axis=0)
    else:
        for i in range(1, 12):
            exec(f'yt_dir.append(np.full(dt_{i}.shape[0],i))')
    if yt_dir:
        Y_test = np.concatenate((yt_dir), axis=0)

    print("[INFO]\tData Load Finished")
    if train_path:
        return torch.tensor(X_train,dtype=torch.float32), torch.tensor(X_test,dtype=torch.float32), Y_train, Y_test
    else:
        return torch.tensor(X_test,dtype=torch.float32),Y_test


def ratio_multiplier(y):
    # set under resample ratio
    multiplier = {1: 0.1}
    target_stats = Counter(y)
    for key, value in target_stats.items():
        if key in multiplier:
            target_stats[key] = int(value * multiplier[key])
    return target_stats


def data_resample(X_train, y, sample_strategy=0):
    """
    We get four resample strategies:
        0.nothing happen
        1.under resample
        2.up resample
        3.under resample + up resample
    """
    if sample_strategy == 1:
        under = ClusterCentroids(
            sampling_strategy=ratio_multiplier, random_state=1,
            estimator=MiniBatchKMeans(n_init=1, random_state=1)
        )
        # under=RandomUnderSampler(sampling_strategy=ratio_multiplier,random_state=2)
        x_resampled, y_resampled = under.fit_resample(X_train, y)

    elif sample_strategy == 2:
        sampling_strategy = Counter(
            {1: 9279, 2: 3099, 3: 3099, 4: 3099, 5: 3099, 6: 3099, 7: 3099, 8: 3099, 9: 3099, 10: 3099, 11: 3099})
        # over1 = BorderlineSMOTE(random_state=1, sampling_strategy=sampling_strategy)
        over2 = SMOTE(random_state=1, sampling_strategy=sampling_strategy)
        # x_resampled, y_resampled = over1.fit_resample(X_train, y)
        x_resampled, y_resampled = over2.fit_resample(X_train, y)

    elif sample_strategy > 2:
        print("resample")
        under = ClusterCentroids(
            sampling_strategy=ratio_multiplier, random_state=1,
            estimator=MiniBatchKMeans(n_init=1, random_state=1)
        )
        x_resampled, y_resampled = under.fit_resample(X_train, y)
        over1 = BorderlineSMOTE(random_state=1)
        over2 = SMOTE(random_state=1)
        x_resampled, y_resampled = over1.fit_resample(x_resampled, y_resampled)
        x_resampled, y_resampled = over2.fit_resample(x_resampled, y_resampled)
    else:
        return X_train, y

    return torch.tensor(x_resampled, dtype=torch.float32), y_resampled


def data_reshape(X,shape):
    pca = PCA(n_components=shape)
    data_2d_pca = pca.fit_transform(X)
    return torch.tensor(data_2d_pca, dtype=torch.float32)


def make_ylabel(y):
    # transform y into multi-label
    Y = np.zeros((y.shape[0], 4))
    for i in range(y.shape[0]):
        if y[i] == 1:
            Y[i] = np.array([1, 0, 0, 0])  # a
        elif y[i] == 2:
            Y[i] = np.array([0, 1, 0, 0])  # c
        elif y[i] == 3:
            Y[i] = np.array([0, 0, 1, 0])  # m
        elif y[i] == 4:
            Y[i] = np.array([0, 0, 0, 1])  # s
        elif y[i] == 5:
            Y[i] = np.array([1, 1, 0, 0])  # ac
        elif y[i] == 6:
            Y[i] = np.array([1, 0, 1, 0])  # am
        elif y[i] == 7:
            Y[i] = np.array([1, 0, 0, 1])  # as
        elif y[i] == 8:
            Y[i] = np.array([0, 1, 1, 0])  # cm
        elif y[i] == 9:
            Y[i] = np.array([1, 1, 1, 0])  # acm
        elif y[i] == 10:
            Y[i] = np.array([1, 1, 0, 1])  # acs
        elif y[i] == 11:
            Y[i] = np.array([1, 1, 1, 1])  # acms
    print("[INFO]\tmulti-label Y:", Y.shape)
    return torch.tensor(Y, dtype=torch.float32)


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = load_data("Data/PSTAAP_train.mat", "Data/PSTAAP_test.mat")
    # print("under resample ratio:", ratio_multiplier((Y_train)))
    # make_ylabel(Y_train)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)