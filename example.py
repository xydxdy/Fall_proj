import numpy as np
import pandas as pd
import glob
from tslearn.preprocessing import TimeSeriesResampler
from scipy import signal
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from pyts.multivariate.transformation import WEASELMUSE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from scipy.signal import medfilt
from tqdm import tqdm
import matplotlib.pyplot as plt

main_path = 'SisFall_dataset/'
samp_rate = 200
n_timestamps = 36000
sensor = ["XAD", "YAD", "ZAD", "XR", "YR", "ZR", "XM", "YM", "ZM"]
chosen = ["XAD", "ZAD", "XR", "ZR"]

def get_data(path):
    """
    read and processing data
    return data (ndarray) shape = (n_features, n_timestamps)
    """
    df = pd.read_csv(path, delimiter=',', header=None)
    df.columns = sensor
    df['ZM'] = df['ZM'].replace({';': ''}, regex=True)
    data = df[chosen].values.T # shape = (n_features, n_timestamps)
    si = (data.shape[-1] // 200) * samp_rate
    data = signal.resample(x=data, num=si, axis=1)
    data = np.pad(data, ((0, 0), (0, n_timestamps-data.shape[-1])), 'constant') # pad zero
    data = medfilt(data, kernel_size=(1,3))
    # data = medfilt(data, kernel_size=1)
    return data # shape = (n_features, n_timestamps)

def get_meta(path):
    """
    get list of metadata from each file
    """
    f = path.split('/')[-1].replace('.txt', '') # D01_SA01_R01
    activity, subject, record = f.split('_') # [D01, SA01, R01]
    label = activity[0] # A or D
    return [label, activity, subject, record]


def load_dataset():
    path_list = glob.glob(main_path+'*/*.txt')
    X, y, meta = [], [], []
    
    for path in tqdm(path_list):
        data_ = get_data(path)
        meta_ = get_meta(path)
        
        X.append(data_)
        y.append(meta_[0])
        meta.append(meta_)
        
    return np.array(X), np.array(y), np.array(meta)

if __name__ == "__main__":
    X, y, meta = load_dataset() 
    """
    X shape =shape = (n_samples, n_features, n_timestamps)

    y = meta[:,0]
    activities = meta[:,1]
    subjects = meta[:,2]
    records = meta[:,3]
    """
    subjects_id = np.unique(meta[:,2])
    SA_id = subjects_id[:23]
    SE_id = subjects_id[23:]

    print(X.shape, y.shape, meta.shape)
    print('\n', subjects_id, '\n', SA_id, '\n', SE_id)
    
    """
    # leave-one-subject-out + StratifiedKFold
    
    for test_subj in SA_id: # leave-one-subject-out
        print('\n===================================')
        print('test subject:', test_subj)
        learn_idxs = np.where(meta[:,2] != test_subj)[0] # list of learning index
        test_idxs = np.where(meta[:,2] == test_subj)[0] # list of test index
        X_learn, y_learn, meta_learn = X[learn_idxs], y[learn_idxs], meta[learn_idxs]
        X_test, y_test, meta_test = X[test_idxs], y[test_idxs], meta[test_idxs]
        
        cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
        activity_learn = meta_learn[:,1]
        for train_idxs, val_idxs in cv.split(X_learn, activity_learn):
            X_train, y_train, meta_train = X_learn[train_idxs], y_learn[train_idxs], meta_learn[train_idxs]
            X_val, y_val, meta_val = X_learn[val_idxs], y_learn[val_idxs], meta_learn[val_idxs]
            
            print('\nX_train:', X_train.shape, 'X_val:', X_val.shape, ', X_test:', X_test.shape)
            
            a_train = np.unique(meta_train[:,1], return_counts=True)
            print('activity_train', dict(zip(a_train[0], a_train[1])))
            
            a_val = np.unique(meta_val[:,1], return_counts=True)
            print('activity_val', dict(zip(a_val[0], a_val[1])))
            
            a_test = np.unique(meta_test[:,1], return_counts=True)
            print('activity_test', dict(zip(a_test[0], a_test[1])))
    """
    
    # leave-one-subject-out + StratifiedShuffleSplit
    for test_subj in SA_id: # leave-one-subject-out
        print('test subject:', test_subj)
        train_idxs = np.where(meta[:,2] != test_subj)[0] # list of train index
        test_idxs = np.where(meta[:,2] == test_subj)[0] # list of test index
        X_train, y_train, meta_train = X[train_idxs], y[train_idxs], meta[train_idxs]
        X_test, y_test, meta_test = X[test_idxs], y[test_idxs], meta[test_idxs]

        muse = WEASELMUSE( )
        logistic  = LogisticRegression()

        pipe = Pipeline(steps=[("muse", muse ), ("logistic", logistic)])

        param_grid = {"mu se__word_size": [2, 4, 6]}
        
        sss = StratifiedShuffleSplit(n_splits=10, random_state=42)
        model = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose = 3, cv = sss)