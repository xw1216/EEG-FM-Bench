import os
import sys
from collections import Counter

import scipy
from scipy import stats
import scipy.io as sio
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from einops import rearrange


current_path = os.path.abspath('./')
root_path = current_path # os.path.split(current_path)[0]

sys.path.append(root_path)

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def EA(x, new_R=None):
    # print(x.shape)
    """
    The Eulidean space alignment approach for EEG data.

    Arg:
        x:The input data,shape of NxCxS
        new_R：The reference matrix.
    Return:
        The aligned data.
    """

    xt = np.transpose(x, axes=(0, 2, 1))
    # print('xt shape:',xt.shape)
    E = np.matmul(x, xt)
    # print(E.shape)
    R = np.mean(E, axis=0)
    # print('R shape:',R.shape)

    R_mat = scipy.linalg.fractional_matrix_power(R, -0.5)
    new_x = np.einsum('n c s,r c -> n r s', x, R_mat)
    if new_R is None:
        return new_x

    new_x = np.einsum('n c s,r c -> n r s', new_x, scipy.linalg.fractional_matrix_power(new_R, 0.5))

    return new_x

        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def select_devices(num_device,gpus=None):
    if gpus is None:
        gpus = torch.cuda.device_count()  
        gpus = [i for i in range(gpus)]
        
    res = []
    last_id = 0
            
    min_memory = 25447170048 // 2  
    for i in range(num_device):
        device_id = gpus[last_id%len(gpus)]
        last_id+=1
        while torch.cuda.get_device_properties(device_id).total_memory < min_memory:
            device_id = gpus[last_id%len(gpus)]
            last_id+=1
        res.append(torch.device(f'cuda:{device_id}') )
    return res

def select_free_gpu():
    gpus = torch.cuda.device_count()  
    if gpus == 0:  
        return None  
    else:  
        device_id = 0  
        min_memory = 25447170048 // 2  
        while True:
            i = random.randint(0, gpus-1)
        # for i in range(gpus):  
            mem_info = torch.cuda.get_device_properties(i)  
            # print(mem_info.total_memory)
            if mem_info.total_memory > min_memory:  
                device_id = i  
                break

        return torch.device(f'cuda:{device_id}') 

def rand_mask(feature):

    for _ in range(np.random.randint(0,4)):
        c = np.random.randint(0,22)

        a = np.random.normal(1,0.4,1)[0]

        feature[:,c] *=a
    return feature

def rand_cov(x):
    # print('xt shape:',xt.shape)
    E = torch.matmul(x, x.transpose(1,2))
    # print(E.shape)
    R = E.mean(0)
    
    U, S, V = torch.svd(R)
    R_mat = U@torch.diag(torch.rand(S.shape[0])*2)@V
    new_x = torch.einsum('n c s,r c -> n r s',x,R_mat)
    return new_x


def shuffle_data(dataset):
    x = rearrange(dataset.x,'(n i) c s->n i c s',n=16)
    y = rearrange(dataset.y,'(n i)->n i',n=16)
    new_x = []
    new_y = []

    for i in np.random.permutation(x.shape[0]):
        index = np.random.permutation(x.shape[1])
        new_x.append(x[i][index])
        new_y.append(y[i][index])

    new_x = torch.stack(new_x)
    new_y = torch.stack(new_y)
    new_x = rearrange(new_x,'a b c d->(a b) c d')
    new_y = rearrange(new_y,'a b->(a b)')

    return eeg_dataset(new_x,new_y)


def print_log(s,path="log.txt"):
    with open(path,"a+") as f:
        f.write((str(s) if type(s) is not str else s) +"\n")
def callback(res):
        print('<Process%s> subject %s accu %s' %(os.getpid(),res['sub'], str(res["accu"])))
        
        
def geban(batch_size=10, n_class=4):
    res = [random.randint(0, batch_size) for i in range(n_class-1) ]
    res.sort()
    # print(res)
    ret=[]
    last=0
    for r in res:
        ret.append(r-last)
        last=r
    ret.append(batch_size-last)
    return ret

def geban_entropy(batch_size=10, n_class=4, entropy_scope=None):
    if entropy_scope is None:
        entropy_scope = [0, 1]
    while True:
        num_class = geban(batch_size, n_class)
        total = sum(num_class)
        ent = stats.entropy([x/total for x in num_class], base=n_class)
        if entropy_scope[0] <= ent <= entropy_scope[1]: break
    return num_class

def sample(batch_size=10, n_class=4):
    res = [random.randint(0, n_class-1) for i in range(batch_size) ]
    res = Counter(res)
    ret = []
    for i in range(n_class):
        ret.append(res[i])
    return ret

def temporal_interpolation(x, desired_sequence_length, mode='nearest', use_avg=True):
    # print(x.shape)
    # squeeze and unsqueeze because these are done before batching
    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)
    if len(x.shape) == 2:
        return torch.nn.functional.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
    # Supports batch dimension
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
    else:
        raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")

class eeg_dataset(Dataset):
    '''
    A class need to input the Dataloader in the pytorch.
    '''
    def __init__(self,feature,label,subject_id=None):
        super(eeg_dataset,self).__init__()

        self.x = feature
        self.y = label
        self.s = subject_id

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def get_num_class(self, num_class=None):
        if num_class is None:
            num_class = [1, 1, 1, 1]
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            label = self.y[i]
            label = int(label)
            if num_class[label]>0:
                num_class[label]-=1
                res[label].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y     
    
    def get_num_subject(self, num_class=None):
        if num_class is None:
            num_class = [1, 1, 1, 1, 1, 1, 1, 1]
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            s = self.s[i]
            s = int(s)
            if num_class[s]>0:
                num_class[s]-=1
                res[s].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y        

    
def get_subj_data(sub, data_path, few_shot_number = 1, is_few_EA = False, target_sample=-1, sess=None, use_average=False):
    
    # target_y_data = []
    
    i=sub
    R=None
    source_train_x = []
    source_train_y = []
    source_valid_x = []
    source_valid_y = []
    
    if sess is not None:
        
        train_path = os.path.join(data_path,r'sub{}_sess{}_train/Data.mat'.format(i, sess))
        test_path = os.path.join(data_path,r'sub{}_sess{}_test/Data.mat'.format(i, sess))
    else:
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        test_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(i))
        
    train_data = sio.loadmat(train_path)
    test_data = sio.loadmat(test_path)
    if use_average:
        train_data['x_data'] = train_data['x_data'] - train_data['x_data'].mean(-2, keepdims=True)
    if is_few_EA is True:
        session_1_x = EA(train_data['x_data'],R)
    else:
        session_1_x = train_data['x_data']

    session_1_y = train_data['y_data'].reshape(-1)

    train_x,valid_x,train_y,valid_y = train_test_split(session_1_x,session_1_y,test_size = 0.1,stratify = session_1_y)
    
    source_train_x.extend(train_x)
    source_train_y.extend(train_y)
    
    source_valid_x.extend(valid_x)
    source_valid_y.extend(valid_y)
    if use_average:
        test_data['x_data'] = test_data['x_data'] - test_data['x_data'].mean(-2, keepdims=True)
        
    if is_few_EA is True:
        session_2_x = EA(test_data['x_data'],R)
    else:
        session_2_x = test_data['x_data']

    session_2_y = test_data['y_data'].reshape(-1)

    train_x,valid_x,train_y,valid_y = train_test_split(session_2_x,session_2_y,test_size = 0.1,stratify = session_2_y)
    
    source_train_x.extend(train_x)
    source_train_y.extend(train_y)
    
    source_valid_x.extend(valid_x)
    source_valid_y.extend(valid_y)
    
    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))
    

    source_valid_x = torch.FloatTensor(np.array(source_valid_x))
    source_valid_y = torch.LongTensor(np.array(source_valid_y))
    
    
    if target_sample>0:
        source_train_x = temporal_interpolation(source_train_x, target_sample)
        source_valid_x = temporal_interpolation(source_valid_x, target_sample)
        
    train_dataset = eeg_dataset(source_train_x,source_train_y)
    valid_datset  = eeg_dataset(source_valid_x,source_valid_y)
    
    return train_dataset, valid_datset

def get_data(sub,data_path,few_shot_number = 1, is_few_EA = False, target_sample=-1, use_avg=True, use_channels=None):
    
    target_session_1_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))
    target_session_2_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(sub))

    session_1_data = sio.loadmat(target_session_1_path)
    session_2_data = sio.loadmat(target_session_2_path)
    R = None
    if is_few_EA is True:
        session_1_x = EA(session_1_data['x_data'],R)
    else:
        session_1_x = session_1_data['x_data']
        
    if is_few_EA is True:
        session_2_x = EA(session_2_data['x_data'],R)
    else:
        session_2_x = session_2_data['x_data']
    
    # -- debug for BCIC 2b
    test_x_1 = torch.FloatTensor(session_1_x)      
    test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)

    test_x_2 = torch.FloatTensor(session_2_x)      
    test_y_2 = torch.LongTensor(session_2_data['y_data']).reshape(-1)
    
    if target_sample>0:
        test_x_1 = temporal_interpolation(test_x_1, target_sample, use_avg=use_avg)
        test_x_2 = temporal_interpolation(test_x_2, target_sample, use_avg=use_avg)
    if use_channels is not None:
        test_dataset = eeg_dataset(torch.cat([test_x_1,test_x_2],dim=0)[:,use_channels,:],torch.cat([test_y_1,test_y_2],dim=0))
    else:
        test_dataset = eeg_dataset(torch.cat([test_x_1,test_x_2],dim=0),torch.cat([test_y_1,test_y_2],dim=0))

    source_train_x = []
    source_train_y = []
    source_train_s = []
    
    source_valid_x = []
    source_valid_y = []
    source_valid_s = []
    subject_id = 0
    for i in range(1,10):
        if i == sub:
            continue
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        train_data = sio.loadmat(train_path)
    
        test_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(i))
        test_data = sio.loadmat(test_path)
        if is_few_EA is True:
            session_1_x = EA(train_data['x_data'],R)
        else:
            session_1_x = train_data['x_data']

        session_1_y = train_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_1_x,session_1_y,test_size = 0.1,stratify = session_1_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)

        if is_few_EA is True:
            session_2_x = EA(test_data['x_data'],R)
        else:
            session_2_x = test_data['x_data']

        session_2_y = test_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_2_x,session_2_y,test_size = 0.1,stratify = session_2_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)
        subject_id+=1
    
    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))
    source_train_s = torch.cat(source_train_s, dim=0)

    source_valid_x = torch.FloatTensor(np.array(source_valid_x))
    source_valid_y = torch.LongTensor(np.array(source_valid_y))
    source_valid_s = torch.cat(source_valid_s, dim=0)
    
    if target_sample>0:
        source_train_x = temporal_interpolation(source_train_x, target_sample, use_avg=use_avg)
        source_valid_x = temporal_interpolation(source_valid_x, target_sample, use_avg=use_avg)
        
    if use_channels is not None:
        train_dataset = eeg_dataset(source_train_x[:,use_channels,:],source_train_y,source_train_s)
    else:
        train_dataset = eeg_dataset(source_train_x,source_train_y,source_train_s)
    
    if use_channels is not None:
        valid_datset = eeg_dataset(source_valid_x[:,use_channels,:],source_valid_y,source_valid_s)
    else:
        valid_datset = eeg_dataset(source_valid_x,source_valid_y,source_valid_s)
    
    return train_dataset,valid_datset,test_dataset


def get_data_Nakanishi2015(sub,data_path="Data/Nakanishi2015_8_64HZ/",few_shot_number = 1, is_few_EA = False, target_sample=-1):
    
    target_session_1_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))

    session_1_data = sio.loadmat(target_session_1_path)
    R = None
    if is_few_EA is True:
        session_1_x = EA(session_1_data['x_data'],R)
    else:
        session_1_x = session_1_data['x_data']
    
    # -- debug for BCIC 2b
    test_x_1 = torch.FloatTensor(session_1_x)      
    test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)
    
    if target_sample>0:
        test_x_1 = temporal_interpolation(test_x_1, target_sample)
        
    test_dataset = eeg_dataset(test_x_1,test_y_1)

    source_train_x = []
    source_train_y = []
    source_train_s = []
    
    source_valid_x = []
    source_valid_y = []
    source_valid_s = []
    subject_id = 0
    for i in tqdm(range(1,11)):
        if i == sub:
            continue
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        # print(train_path)
        train_data = sio.loadmat(train_path)
        
        if is_few_EA is True:
            session_1_x = EA(train_data['x_data'],R)
        else:
            session_1_x = train_data['x_data']

        session_1_y = train_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_1_x,session_1_y,test_size = 40,stratify = session_1_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)
        subject_id+=1
    
    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))
    source_train_s = torch.cat(source_train_s, dim=0)

    source_valid_x = torch.FloatTensor(np.array(source_valid_x))
    source_valid_y = torch.LongTensor(np.array(source_valid_y))
    source_valid_s = torch.cat(source_valid_s, dim=0)
    
    if target_sample>0:
        source_train_x = temporal_interpolation(source_train_x, target_sample)
        source_valid_x = temporal_interpolation(source_valid_x, target_sample)
        
    train_dataset = eeg_dataset(source_train_x,source_train_y,source_train_s)
    
    valid_datset = eeg_dataset(source_valid_x,source_valid_y,source_valid_s)
    
    return train_dataset,valid_datset,test_dataset

def get_data_Wang2016(sub,data_path="Data/Wang2016_4_20HZ/",few_shot_number = 1, is_few_EA = False, target_sample=-1):
    
    target_session_1_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))

    session_1_data = sio.loadmat(target_session_1_path)
    R = None
    if is_few_EA is True:
        session_1_x = EA(session_1_data['x_data'],R)
    else:
        session_1_x = session_1_data['x_data']
    
    # -- debug for BCIC 2b
    test_x_1 = torch.FloatTensor(session_1_x)      
    test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)
    
    if target_sample>0:
        test_x_1 = temporal_interpolation(test_x_1, target_sample)
        
    test_dataset = eeg_dataset(test_x_1,test_y_1)

    source_train_x = []
    source_train_y = []
    source_train_s = []
    
    source_valid_x = []
    source_valid_y = []
    source_valid_s = []
    subject_id = 0
    for i in tqdm(range(1,36)):
        if i == sub:
            continue
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        # print(train_path)
        train_data = sio.loadmat(train_path)
        
        if is_few_EA is True:
            session_1_x = EA(train_data['x_data'],R)
        else:
            session_1_x = train_data['x_data']

        session_1_y = train_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_1_x,session_1_y,test_size = 40,stratify = session_1_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)
        subject_id+=1
    
    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))
    source_train_s = torch.cat(source_train_s, dim=0)

    source_valid_x = torch.FloatTensor(np.array(source_valid_x))
    source_valid_y = torch.LongTensor(np.array(source_valid_y))
    source_valid_s = torch.cat(source_valid_s, dim=0)
    
    if target_sample>0:
        source_train_x = temporal_interpolation(source_train_x, target_sample)
        source_valid_x = temporal_interpolation(source_valid_x, target_sample)
        
    train_dataset = eeg_dataset(source_train_x,source_train_y,source_train_s)
    
    valid_datset = eeg_dataset(source_valid_x,source_valid_y,source_valid_s)
    
    return train_dataset,valid_datset,test_dataset
if __name__=="__main__":
    train_dataset,valid_dataset,test_dataset = get_data_Wang2016(1,"Data/Wang2016_4_20HZ/", 1, is_few_EA = False, target_sample=-1)
    # # train_dataset,valid_dataset,test_dataset = get_data(1,data_path,1,True)
    # avg_ent = 0 

    # for i in range(1000):
    #     # print(geban()) 
    #     # print(sample())
    #     num_class = geban_entropy(entropy_scope=[1.2,1e6])#geban()#sample()
    #     total = sum(num_class)
    #     num_class = [x/total for x in num_class]
    #     # print(num_class)
    #     # print(sum([-x*(math.log(x)) for x in num_class if x>0]))
    #     ent = stats.entropy(num_class) 
    #     avg_ent+=ent
    #     print(avg_ent/1000) # sample 1.2110981470145854 geban 0.9734407215366253
    
    
import mne
import torch
import tqdm
import pandas as pd 
import csv
import numpy as np
import os
import scipy.io as scio

import random
mne.set_log_level("ERROR")

def min_max_normalize(x: torch.Tensor, data_max=None, data_min=None, low=-1, high=1):
    if data_max is not None:
        max_scale = data_max - data_min
        scale = 2 * (torch.clamp_max((x.max() - x.min()) / max_scale, 1.0) - 0.5)
        
    if len(x.shape) == 2:
        xmin = x.min()
        xmax = x.max()
        if xmax - xmin == 0:
            x = 0
            return x
    elif len(x.shape) == 3:
        xmin = torch.min(torch.min(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        xmax = torch.max(torch.max(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        constant_trials = (xmax - xmin) == 0
        if torch.any(constant_trials):
            # If normalizing multiple trials, stabilize the normalization
            xmax[constant_trials] = xmax[constant_trials] + 1e-6

    x = (x - xmin) / (xmax - xmin)

    # Now all scaled 0 -> 1, remove 0.5 bias
    x -= 0.5
    # Adjust for low/high bias and scale up
    x += (high + low) / 2
    x  = (high - low) * x
    
    if data_max is not None:
        x = torch.cat([x, torch.ones((1, x.shape[-1])).to(x)*scale])
    return x
    
    
use_channels_names=[
               'FP1', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
                'O1', 'O2'
    ]

# -- read Kaggle ERN
ch_names_kaggle_ern = list("Fp1,Fp2,AF7,AF3,AF4,AF8,F7,F5,F3,F1,Fz,F2,F4,F6,F8,FT7,FC5,FC3,FC1,FCz,FC2,FC4,FC6,FT8,T7,C5,C3,C1,Cz,C2,C4,C6,T8,TP7,CP5,CP3,CP1,CPz,CP2,CP4,CP6,TP8,P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,POz,PO8,O1,O2".split(','))

def read_csv_epochs(filename, tmin, tlen, use_channels_names=use_channels_names, data_max=None, data_min=None):
    sample_rate = 200
    raw = pd.read_csv(filename)
    
    data = torch.tensor(raw.iloc[:,1:-2].values) # exclude time EOG Feedback
    feed = torch.tensor(raw['FeedBackEvent'].values)
    stim_pos = torch.nonzero(feed>0)
    # print(stim_pos)
    datas = []
    
    # -- get channel id by use chan names
    if use_channels_names is not None:
        choice_channels = []
        for ch in use_channels_names:
            choice_channels.append([x.lower().strip('.') for x in ch_names_kaggle_ern].index(ch.lower()))
        use_channels = choice_channels
    if data_max is not None: use_channels+=[-1]
    
    xform = lambda x: min_max_normalize(x, data_max, data_min)
    
    for fb, pos in enumerate(stim_pos, 1):
        start_i = max(pos + int(sample_rate * tmin), 0)
        end___i = min(start_i + int(sample_rate * tlen), len(feed))
        # print(start_i, end___i)
        trial = data[start_i:end___i, :].clone().detach().cpu().numpy().T
        # print(trial.shape)
        info = mne.create_info(
            ch_names=[str(i) for i in range(trial.shape[0])],
            ch_types="eeg",  # channel type
            sfreq=200,  # frequency
            #
        )
        raw = mne.io.RawArray(trial, info)  # create raw
        # raw = raw.filter(5,40)
        # raw = raw.resample(256)
        
        trial = torch.tensor(raw.get_data()).float()

        trial = xform(trial)
        if use_channels_names is not None:
            trial = trial[use_channels]
        datas.append(trial)
    return datas
    
def read_kaggle_ern_test(
                    path     = "datas/",
                    subjects = [1,3,4,5,8,9,10,15,19,25],#
                    sessions = [1,2,3,4,5],#
                    tmin     = -0.7,
                    tlen     = 2,
                    data_max=None,
                    data_min=None,
                    use_channels_names = use_channels_names,
                    ):
    # -- read labels
    labels = pd.read_csv(os.path.join(path, 'KaggleERN', 'true_labels.csv'))['label']
    
    # -- read datas
    label_id = 0
    datas = []
    for i in tqdm.tqdm(subjects):
        for j in sessions:
            filename = os.path.join(path, "KaggleERN", "test", "Data_S{:02d}_Sess{:02d}.csv".format(i,j))

            # -- read data
            for data in read_csv_epochs(filename, tmin=tmin, tlen=tlen, data_max=data_max, data_min=data_min, use_channels_names = use_channels_names): 
                label = labels[label_id]
                label_id += 1
                datas.append((data, int(label)))
    return datas

def read_kaggle_ern_train(
                    path     = "datas/",
                    subjects = [2,6,7,11,12,13,14,16,17,18,20,21,22,23,24,26, ],#
                    sessions = [1,2,3,4,5],#
                    tmin     = -0.7,
                    tlen     = 2,
                    data_max=None,
                    data_min=None,
                    use_channels_names = use_channels_names,
                    ):
    # -- read labels
    labels = []
    with open(os.path.join(path, 'KaggleERN', 'TrainLabels.csv'), 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if(i>0): labels.append(row)
    labels = dict(labels) # [['S02_Sess01_FB001', '1'],
    
    # -- read datas
    datas = []
    for i in tqdm.tqdm(subjects):
        for j in sessions:
            if i>9:
                if(i == 22 and j == 5):
                    print("Skipped error file " + "KagglERN/train/Data_S"+ str(i)+"_Sess0"+str(j)+".csv" )
                else:
                    filename = os.path.join(path, "KaggleERN","train","Data_S"+ str(i)+"_Sess0"+str(j)+".csv")
            else:
                filename = os.path.join(path, "KaggleERN", "train", "Data_S0"+ str(i)+"_Sess0"+str(j)+".csv")
            
            # -- read data
            for fb,trial in enumerate(read_csv_epochs(filename, tmin=tmin, tlen=tlen, data_max=data_max, data_min=data_min, use_channels_names = use_channels_names),1): 
                label = labels["S{:02d}_Sess{:02d}_FB{:03d}".format(i,j,fb)]
                datas.append((trial, int(label)))
    return datas

#  chs: 25 EEG
#  custom_ref_applied: False
#  highpass: 0.5 Hz
#  lowpass: 100.0 Hz
#  meas_date: 2005-01-19 12:00:00 UTC
#  nchan: 25
#  projs: []
#  sfreq: 250.0 Hz
if __name__=="__main__":
    # datas = read_edf_epochs('datas\\PhysioNetMI\\S001\\S001R04.edf')
    # print(datas[0])
    # subject_data = read_physionetmi()
    # print(len(subject_data))
    path = "/path/to/your/datasets/backup"
    data = read_kaggle_ern_test(path, subjects=[1], sessions=[1])
    print(data[0][0].shape)
    # read_kaggle_ern_train(path)