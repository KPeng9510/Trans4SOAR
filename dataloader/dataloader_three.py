import numpy as np

from torch.utils.data import Dataset
import tools
import tools_dataset
import pickle as pkl
import torch
ntu_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)
)



#@DATASETS.register_module()

class Feeder(Dataset):
    def __init__(self, data_path=None, label_path=None, p_interval=1, mode='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False, transforms=None):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.mode = mode
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = True
        self.vel = True
        self.transforms = transforms
        
        f = open('/home/haicore-project-kit-iar-cvhci/fy2374//occ_vis/data_info_03.pkl', 'rb')
        self.data_info = pkl.load(f)
        f.close()                  
        f = open('/home/haicore-project-kit-iar-cvhci/fy2374//occ_vis/key_info_03.pkl', 'rb')
        self.key_info = pkl.load(f)
        f.close()
        f = open('/home/haicore-project-kit-iar-cvhci/fy2374//occ_vis/num_info_03.pkl', 'rb')
        self.num_info = pkl.load(f)
        f.close()
        
        f = open('/home/haicore-project-kit-iar-cvhci/fy2374//occ/ntu120train.pkl', 'rb')
        train_info = pkl.load(f)
        f.close()                  
        f = open('/home/haicore-project-kit-iar-cvhci/fy2374//occ/ntu120test.pkl', 'rb')
        test_info = pkl.load(f)
        f.close()
        f = open('/home/haicore-project-kit-iar-cvhci/fy2374//occ/ntu120val.pkl', 'rb')
        val_info = pkl.load(f)
        f.close()
        f = open('/home/haicore-project-kit-iar-cvhci/fy2374//occ_vis/length_03.pkl', 'rb')
        self.length_info = pkl.load(f)
        f.close()
        self.key_info = [item.split('/')[-1] for item in self.key_info]
        self.activity_info = [item.split('A')[-1].split('.')[0] for item in self.key_info] 
        train_info = [item.split('/')[-1] for item in train_info]
        test_info = [item.split('/')[-1] for item in test_info]
        val_info = [item.split('/')[-1] for item in val_info]
        print(len(self.key_info))
        
        if mode == 'train':
            f = open('/home/haicore-project-kit-iar-cvhci/fy2374//occ/train_index.pkl', 'rb')
            self.index_list = pkl.load(f)
            f.close() 
            #self.index_list = [self.key_info.index(item) for item in train_info] 
        elif mode == 'val':
            f = open('/home/haicore-project-kit-iar-cvhci/fy2374//occ/val_index.pkl', 'rb')
            self.index_list = pkl.load(f)
            f.close() 
            #self.index_list = [self.key_info.index(item) for item in val_info]
        else:
            f = open('/home/haicore-project-kit-iar-cvhci/fy2374//occ/test_index.pkl', 'rb')
            self.index_list = pkl.load(f)
            f.close() 
            #self.index_list = [self.key_info.index(item) for item in test_info]
        self.targets = [int(self.activity_info[item])-1 for item in self.index_list]
        self.target_remap()

    def target_remap(self,):
        classes = list(np.sort(np.unique(np.array(self.targets))))
        #print(classes)
        for idx, i in enumerate(classes):
            mask = np.array(self.targets) == i
            self.targets = np.array(self.targets)
            self.targets[mask] = idx
            self.targets = list(self.targets)
        print(len(np.unique(self.targets)))

    def random_masked(self, joints):
        #batch,3,256,256
        joints = torch.Tensor(joints)
        import random
        r=random.sample([0.001,0.002,0.003,0.004], 1)[0]
        p= random.sample([1,2,3],1)[0]
        q = 1
        T,N,C = joints.size()
        if T//p !=0:
            joints = torch.cat([joints, torch.zeros([p-T%p,N,C])],dim=0)
        t,n,c = joints.size()
        num_patches = (t//p)*(n//q)
        masked_ind = random.sample(range(0,num_patches), int(r*num_patches))
        joints = joints.contiguous().view(t//p,p,n//q,q,c).permute(0,2,1,3,4).contiguous().view(num_patches,p,q,c)
        joints[masked_ind,...] = torch.zeros_like(joints[masked_ind,...])
        joints = joints.contiguous().view(t//p,n//q,p,q,c).permute(0,2,1,3,4).contiguous().view(t,n,c)
        return joints[:T,:,:].numpy()

    def random_temporal_mask(self, joints):
        import random
        joints = torch.Tensor(joints)
        t,n,c = joints.shape
        r = random.sample([0.1,0.2], 1)
        num = int(r[0]*t*n)
        masked_ind =random.sample(range(0,t*n), num)
        joints = joints.flatten(0,1)
        joints[masked_ind,...] = torch.zeros_like(joints[masked_ind,...])
        return joints.contiguous().view(t,n,c).numpy()
    
    def __len__(self):
        return len(self.index_list)

    def __iter__(self):
        return self
    def preprocess(self, data, length):
        t,c = data.shape

        data = data.reshape(t//25,25,3)
        body1=data[:length]
        if data.shape[0] == length:
            body2=np.zeros_like(body1)
            data_all = np.concatenate((body1, body2),axis=1)
        else:
            body2=data[length:]
            if (data.shape[0] - length) >= length:
                container = np.zeros_like(body2)
                container[:length] = body1
                data_all = np.concatenate((container, body2),axis=1)
            else:
                container = np.zeros_like(body1)
                container[:body2.shape[0]] = body2
                data_all = np.concatenate((body1, container),axis=1)
        return data_all
    def preprocess_norm(self, data, length):
        return data
        
    def get_doubtful_region(self, data):
        T,N,C = data.shape
        data = data.reshape(T*N, C)
        mask = data.sum(-1) == 0
        if (mask.sum() != 0) & (data.sum()!=0):
            mean = data[~mask,:].mean(0)
            data[mask,:] = np.repeat(np.expand_dims(mean,axis=0), data[mask,:].shape[0],0)
        return data.reshape(T,N, C)#, d_mask
    def __getitem__(self, ind):
        index = self.index_list[ind]
        self.bone=True
        self.vel=True
        data_numpy = self.data_info[index]
        length = self.length_info[index]
        label = self.targets[ind]
        key = self.key_info[index]
        name = key.split('/')[-1].split('.')[0]
        data_numpy = self.preprocess(data_numpy, length) # t,m,n,c
        T,N,C = data_numpy.shape
        #if self.mode !='val' and self.mode!='test':
        #    data_numpy=self.random_temporal_mask(data_numpy)
        data_numpy = self.get_doubtful_region(data_numpy)
        if self.bone:
            data_numpyk = data_numpy.reshape(T,25,N//25,C)
            bone_data_numpy = np.zeros_like(data_numpyk)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, v1 - 1] = data_numpyk[:, v1 - 1] - data_numpyk[:, v2 - 1]
            bone_data_numpy = bone_data_numpy.reshape(T,N,C)
        if self.vel:
            vel_data_numpy = np.zeros_like(data_numpy)
            vel_data_numpy[:-1,:] = data_numpy[1:,:] - data_numpy[:-1,:]
            vel_data_numpy[-1,:] = 0
            #vel_data_numpy = self.preprocess_norm(vel_data_numpy, length)
        data_numpy = torch.Tensor(np.resize(data_numpy,(256,256,3)).astype(np.float64)).permute(2,0,1) #self.transforms(data_numpy)
        bone_data_numpy = torch.Tensor(np.resize(bone_data_numpy,(256,256,3))).permute(2,0,1)#self.transforms(bone_data_numpy)
        vel_data_numpy = torch.Tensor(np.resize(vel_data_numpy,(256,256,3))).permute(2,0,1) #self.transforms(vel_data_numpy)
 
        return torch.stack([data_numpy, bone_data_numpy, vel_data_numpy], axis=0), torch.Tensor([label]).long()

