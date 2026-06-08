import os
import pickle
import torch
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from typing import List

torch.manual_seed(0)
np.random.seed(0)

def _fit_or_load_scalers(data_views: List[np.ndarray],
                         db_name: str,
                         path: str,
                         use_cache: bool = True):

    scaler_path = os.path.join(path, f'{db_name}_scaler.pkl')
    scalers = []

    if use_cache and os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
    else:
        for v in range(len(data_views)):
            scaler = StandardScaler().fit(data_views[v])
            scalers.append(scaler)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers, f)

    # 变换
    for v in range(len(data_views)):
        data_views[v] = scalers[v].transform(data_views[v])
    return data_views


class MultiviewData(Dataset):
    def __init__(self,
                 db: str,
                 device: torch.device,
                 path: str = "datasets/",
                 training: bool = True,
                 use_cache: bool = True,
                 ):
        super().__init__()
        self.device = device
        self.training = training
        self.data_views: List[np.ndarray] = []
        self.labels = None
        self.num_views = 0

        # ---------- 读 .mat ----------
        if db == "MSRCv1":
            mat = sio.loadmat(os.path.join(path, 'MSRCv1.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            # self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
            self.labels = self.labels - self.labels.min()

        elif db == "ORL":
            mat = sio.loadmat(os.path.join(path, 'ORL.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "MNIST-USPS":
            mat = sio.loadmat(os.path.join(path, 'MNIST_USPS.mat'))
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)
            self.data_views.extend([X1.reshape(X1.shape[0], -1),
                                    X2.reshape(X2.shape[0], -1)])
            self.num_views = 2
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "BDGP":
            mat = sio.loadmat(os.path.join(path, 'BDGP.mat'))
            self.data_views.extend([mat['X1'].astype(np.float32),
                                    mat['X2'].astype(np.float32)])
            self.num_views = 2
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "Fashion":
            mat = sio.loadmat(os.path.join(path, 'Fashion.mat'))
            X1 = mat['X1'].reshape(mat['X1'].shape[0], -1).astype(np.float32)
            X2 = mat['X2'].reshape(mat['X2'].shape[0], -1).astype(np.float32)
            X3 = mat['X3'].reshape(mat['X3'].shape[0], -1).astype(np.float32)
            self.data_views.extend([X1, X2, X3])
            self.num_views = 3
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "COIL20":
            mat = sio.loadmat(os.path.join(path, 'COIL20.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            # self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
            self.labels = self.labels - self.labels.min()

        elif db == "hand":
            mat = sio.loadmat(os.path.join(path, 'handwritten.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "scene":
            mat = sio.loadmat(os.path.join(path, 'Scene15.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "NUSWIDEOBJ":
            mat = sio.loadmat(os.path.join(path, 'NUSWIDEOBJ.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            # self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
            self.labels = self.labels - self.labels.min()

        elif db == "cifar10":
            mat = sio.loadmat(os.path.join(path, 'cifar10.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        else:
            raise NotImplementedError

        # ---------- 统一标准化 ----------
        self.data_views = _fit_or_load_scalers(self.data_views, db, path, use_cache)

        # ---------- 转成 tensor ----------
        for idx in range(self.num_views):
            self.data_views[idx] = torch.from_numpy(self.data_views[idx]).to(device)

        # 打印信息
        for idx in range(self.num_views):
            print(f"View {idx} data shape: {self.data_views[idx].shape}")
        print(f"Labels shape: {self.labels.shape}")
        print(f"Labels distribution: {np.bincount(self.labels)}")


    def _augment(self, data: torch.Tensor) -> torch.Tensor:
        # 更温和的噪声
        noise = torch.randn_like(data) * 0.005
        data = data + noise

        # 更高的保留率
        mask = torch.bernoulli(torch.full(data.shape, 0.95, device=data.device))
        data = data * mask

        return data

    # ---------- Dataset 接口 ----------
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sub_views = []
        for v in range(self.num_views):
            x = self.data_views[v][index]
            if self.training:
                x = self._augment(x)
            sub_views.append(x)
        return sub_views, int(self.labels[index])

    def set_training(self, mode: bool = True):
        self.training = mode


def get_multiview_data(mv_data,
                       batch_size: int,
                       shuffle: bool = True,
                       drop_last: bool = False):

    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return loader, num_views, num_samples, num_clusters


def get_all_multiview_data(mv_data):

    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=num_samples,
        shuffle=True,
        drop_last=False,
    )
    return loader, num_views, num_samples, num_clusters