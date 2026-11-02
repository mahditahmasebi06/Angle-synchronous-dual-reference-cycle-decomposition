"""Read the 236 records of the field incipient-fault dataset (Xiong et al.) into data/xiong.npz.

Clone https://github.com/xiongsiheng/Incipient-fault-waveform-dataset into data/ first
(or set XIONG_DIR). Each record: 1312 samples x 7 channels (Ia, Ib, Ic, In, Va, Vb, Vc) at 4096 Hz."""
import os
import numpy as np
import paths

FS = 4096.0; F0 = 50.0


def load_all(d=paths.XIONG_DIR):
    y = np.loadtxt(f'{d}/label.txt').astype(int)
    ys = np.loadtxt(f'{d}/label_subtype.txt').astype(int)
    X = np.stack([np.loadtxt(f'{d}/waveform/{i + 1}.txt') for i in range(len(y))])
    return X, y, ys


if __name__ == '__main__':
    X, y, ys = load_all()
    os.makedirs(os.path.dirname(paths.DATA_NPZ), exist_ok=True)
    np.savez_compressed(paths.DATA_NPZ, X=X, y=y, ys=ys)
    print(paths.DATA_NPZ, X.shape, np.bincount(y).tolist())
