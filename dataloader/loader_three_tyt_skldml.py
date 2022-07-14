import numpy as np
import torchvision
from torch.utils.data import Dataset
import tools
import tools_dataset
import pickle as pkl
import torch
from PIL import Image
from scipy import misc
ntu_pairs = (
    (0,1),(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(1,8),(1,11),(8,9),(9,10),(11,12)
)
def image_downsample(image):
    shape = image.shape
    #print(len(shape))
    if len(shape) < 4:
        return np.zeros((64, 256, 256, 3))
    frames = image.shape[0]
    step = int(np.floor(frames/64)+1)
    #print(frames)
    #print(step)
    #print(step)
    container = np.zeros((64, image.shape[1], image.shape[2], image.shape[3]))
    for i in range(64):
        if (i*step < frames):
            container[i, :,:,:] = image[i*step, :, :, :]
    return container
#@DATASETS.register_module()
def extract_frames(video_path,  overwrite=False, start=-1, end=-1, every=1):
    """
    Extract frames from a video using OpenCVs VideoCapture
    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :return: count of images saved
    """
    videop=video_path
    video_root = '/cvhci/data/activity/Drive&Act/kunyu/videos/'
    #print(video_path)
    video_path = os.path.normpath(video_root+video_path)  # make the paths OS (Windows) compatible
    #frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible
    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path
    #print(video_path)
    assert os.path.exists(video_path)  # assert the video file exists
    capture = cv2.VideoCapture(video_path)  # open the video using OpenCV
    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(1, start)  # set the starting frame of the capture
    frame = start  # keep track of which frame we are up to, starting from start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    saved_count = 0  # a count of how many frames we have saved
    img_list=[]
    #print(start)
    while frame < end:  # lets loop through the frames until the end
        _, image = capture.read()  # read an image from the capture
        if while_safety > 500:  # break the while if our safety maxs out at 500
            break
        # sometimes OpenCV reads None's during a video, in which case we want to just skip
        if image is None:  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            print('false frame')
            continue  # skip
        if frame % every == 0:  # if this is a frame we want to write out based on the 'every' argument
            while_safety = 0  # reset the safety count
            #save_path = os.path.join(frames_dir, video_filename, "{:010d}.jpg".format(frame))  # create the save path
            #if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
            #    cv2.imwrite(save_path, image)  # save the extracted image
            #    saved_count += 1  # increment our counter by one
            image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            img_list.append(resize(image, (224,224)))
            #if (videop == 'vp14/run1_2018-05-30-10-11-09.ids_1.avi') and (start == 10486):
            #    #img_list.append(resize(image, (224,224)))
            #    print(video_path)
            #    cv2.imwrite('/home/kpeng/driveact/Video-Swin-Transformer/images/eat'+str(frame)+'.png',image)
            #print(image)
        frame += 1  # increment our frame count
    capture.release()  # after the while has finished close the capture
    #v2.imwrite(str(frame)+'.png', ) 
    out = image_downsample(np.array(img_list))
    #print(out.shape)
    #if (videop == 'vp14/run1_2018-05-30-10-11-09.ids_1.avi') and (start == 10486):
    #    sys.exit()
    return out, np.array(img_list)  # and return the count of the images we saved
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
        self.name_list_train = ['Cook.Cleandishes','Cook.Cut','Drink.Fromcan','Drink.Fromglass','Eat.Snack','Getup','Leave','Makecoffee.Pourwater','Maketea.Insertteabag','Pour.Fromkettle','Uselaptop',
        'Usetelephone','Cook.Cleanup','Drink.Frombottle','Drink.Fromcup','Eat.Attable','Enter','Laydown','Makecoffee.Pourgrains','Maketea.Boilwater','Pour.Frombottle','Sitdown','Usetablet','WatchTV']
        self.name_list_val = ['Cook.Stir','Cook.Usestove','Cutbread','Pour.Fromcan','Readbook','Takepills','Walk']
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
        self.resize = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(), torchvision.transforms.Resize([256,256]), torchvision.transforms.ToTensor()])

        f = open('/hkfs/work/workspace/scratch/fy2374-blah/occ_3d/tyt/tytdata_info.pkl', 'rb')
        self.data_info = pkl.load(f)
        f.close()                  
        f = open('/hkfs/work/workspace/scratch/fy2374-blah/occ_3d/tyt/tytkey_info.pkl', 'rb')
        self.key_info = pkl.load(f)
        f.close()
        #f = open('/hkfs/work/workspace/scratch/fy2374-blah/occ_3d/tyt/num_info_03_005_60.pkl', 'rb')
        #self.num_info = pkl.load(f)
        #f.close()
        f = open('/hkfs/work/workspace/scratch/fy2374-blah/occ_3d/tyt/tyttrain.pkl', 'rb')
        train_info = pkl.load(f)
        f.close()                  
        f = open('/hkfs/work/workspace/scratch/fy2374-blah/occ_3d/tyt/tytval.pkl', 'rb')
        test_info = pkl.load(f)
        f.close()
        f = open('/hkfs/work/workspace/scratch/fy2374-blah/occ_3d/tyt/tyttest.pkl', 'rb')
        val_info = pkl.load(f)
        f.close()
        f = open('/hkfs/work/workspace/scratch/fy2374-blah/occ_3d/tyt/tytlength_info.pkl', 'rb')
        self.length_info = pkl.load(f)
        f.close()
        self.activity_info = [item.split('/')[-2] for item in self.key_info]
        self.key_info = [item.split('/')[-1] for item in self.key_info]
        train_info = [item.split('/')[-1] for item in train_info]
        test_info = [item.split('/')[-1] for item in test_info]
        val_info = [item.split('/')[-1] for item in val_info]
        print(len(self.key_info))
        if mode == 'train':
            #f = open('/hkfs/work/workspace/scratch/fy2374-blah/occ_3d/ntu_60/train_index_60.pkl', 'rb')
            self.index_list = list(np.arange(0, len(train_info)))#pkl.load(f)
            #f.close() 
            #self.index_list = [self.key_info.index(item) for item in train_info] 
        elif mode == 'val':
            #f = open('/hkfs/work/workspace/scratch/fy2374-blah/occ_3d/ntu_60/val_index_60.pkl', 'rb')
            self.index_list = list(np.arange(len(train_info), len(train_info)+len(test_info))) #pkl.load(f)
            print(len(self.index_list))
            #f.close() 
            #self.index_list = [self.key_info.index(item) for item in val_info]
        else:
            #f = open('/hkfs/work/workspace/scratch/fy2374-blah/occ_3d/ntu_60/test_index_60.pkl', 'rb')
            self.index_list = list(np.arange(len(train_info)+len(test_info), len(train_info)+len(test_info)+len(val_info)))
            #pkl.load(f)
            print(len(self.index_list))
            #f.close() 
            ##self.index_list = [self.key_info.index(item) for item in test_info]
        if self.mode == 'train':
            self.targets = [self.name_list_train.index(self.activity_info[item]) for item in self.index_list]
        else:
            self.targets = [self.name_list_val.index(self.activity_info[item]) for item in self.index_list]
        #self.targets = [int(self.activity_info[int(item)])-1 for item in self.index_list]
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
        '''
        x_min = np.min(data[:,0])
        x_max = np.max(data[:,0])
        y_min = np.min(data[:,1])
        y_max = np.max(data[:,1])
        z_min = np.min(data[:,2])
        z_max = np.max(data[:,2])
        if x_max != x_min:
            data[:,0] = (data[:,0] - x_min)*255/(x_max - x_min)
        else:
            data[:,0] = np.zeros_like(data[:,0])
        if y_max != y_min:
            data[:,1] = (data[:,1] - y_min)*255/(y_max - y_min)
        else:
            data[:,1] = np.zeros_like(data[:,0])
        if z_max != z_min:
            data[:,2] = (data[:,2] - z_min)*255/(z_max - z_min)
        else:
            data[:,2] = np.zeros_like(data[:,0])
        '''
        data = data.reshape(t//13,13,2)
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
    def random_temporal_mask(self, joints):
        import random
        joints = torch.Tensor(joints)
        t,n,c = joints.shape
        r = random.sample([0.1], 1)
        num = int(r[0]*t*n)
        masked_ind =random.sample(range(0,t*n), num)
        joints = joints.flatten(0,1)
        joints[masked_ind,...] = torch.zeros_like(joints[masked_ind,...])
        return joints.contiguous().view(t,n,c).numpy()
    def preprocess_norm(self, data, length):
        
        x_min = np.min(data[...,0])
        x_max = np.max(data[...,0])
        y_min = np.min(data[...,1])
        y_max = np.max(data[...,1])
        #z_min = np.min(data[...,2])
        #z_max = np.max(data[...,2])

        if x_max != x_min:
            data[...,0] = (data[...,0] - x_min)*255/(x_max - x_min)
        else:
            data[...,0] = np.zeros_like(data[...,0])
        if y_max != y_min:
            data[...,1] = (data[...,1] - y_min)*255/(y_max - y_min)
        #else:
        #    data[...,1] = np.zeros_like(data[...,0])
        #if z_max != z_min:
        #    data[...,2] = (data[...,2] - z_min)*255/(z_max - z_min)
        #else:
        #    data[...,2] = np.zeros_like(data[...,0])
        
        return data
    def get_doubtful_region(self, data):
        #print(data.shape)
        T,N,C = data.shape
        data = data.reshape(T*N, C)
        #d_mask = np.zeros_like(data)
        mask = data.sum(-1) == 0
        if (mask.sum() != 0) & (data.sum()!=0):
            #d_mask[mask,:] == 1
            mean = data[~mask,:].mean(0)
            #print(mean.shape)
            #std = np.std(data[~mask].flatten(0,1), axis=0)
            data[mask,:] = np.repeat(np.expand_dims(mean,axis=0), data[mask,:].shape[0],0) 
        return data.reshape(T,N, C)
    def pil_loader(self, path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)

        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    def __getitem__(self, ind):
        index = self.index_list[ind]
        self.bone=True
        self.vel=True
        data_numpy = self.data_info[index]
        length = self.length_info[index]
        #print(ind)
        #label = self.targets[ind]
        key = self.key_info[index]
        if self.mode=='train':
            label = self.targets[ind]
        else:
            label = self.targets[ind]
        #path_3d = '/export/md0/datasets/NTU_RGBD/heatmap_joints/' + name + '.mp4'
        #video = extract_frames(path_3d)
        #data_numpy = np.array(data_numpy)
        #valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        #data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        #if self.random_rot:
        #    data_numpy = tools.random_rot(data_numpy)
        T,N,C = data_numpy.shape
        data_numpy = data_numpy.reshape(T*N, C)
        data_numpy = self.preprocess(data_numpy, length) # t,m,n,c
        #data_numpy_masked = self.get_doubtful_region(self.random_masked(data_numpy))
        #bone_data_numpy_masked = self.random_masked(bone_data_numpy)
        #vel_data_numpy_masked = self.random_masked(vel_data_numpy)
        #print(data_numpy.shape)
        
        #data_numpy = np.transpose(data_numpy,[1,0,2])
        #print(data_numpy.shape)
        T,N,C = data_numpy.shape
        #N,T,C = image.shape
        '''
        data_numpy = np.transpose(image, (1,0,2))
        if N == 25:
            data_numpy = np.concatenate([image, np.zeros_like(image)], axis=1)
        T,N,C = data_numpy.shape
        '''
        #print(T,N,C)
        #data_numpy = self.preprocess(data_numpy, length)
        #T,N,C = data_numpy.shape
        #if self.mode !='val':
        #    data_numpy = tools_dataset.random_rot(data_numpy.view())
        #data_numpy = self.preprocess(data_numpy, length) # t,m,n,c
        #if self.mode!='val' and self.mode!='test':
        #    data_numpy = self.random_temporal_mask(data_numpy)
        #bone_data_numpy_masked = self.random_masked(bone_data_numpy)
        #vel_data_numpy_masked = self.random_masked(vel_data_numpy)
        #data_numpy = self.get_doubtful_region(data_numpy)
        if self.bone:
            #print(data_numpy.shape)
            data_numpyk = data_numpy.reshape(T,13,N//13,C)
            #from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpyk)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, v1 - 1] = data_numpyk[:, v1 - 1] - data_numpyk[:, v2 - 1]
            #bone_data_numpy = self.preprocess_norm(bone_data_numpy, length)
            bone_data_numpy = bone_data_numpy.reshape(T,N,C)
            
            #bone_data_numpy_masked = np.zeros_like(data_numpy_masked)
            #for v1, v2 in ntu_pairs:
            #    bone_data_numpy_masked[:, v1 - 1] = data_numpy_masked[:, v1 - 1] - data_numpy_masked[:, v2 - 1]
            #bone_data_numpy_masked = self.preprocess_norm(bone_data_numpy_masked, length)
            
        if self.vel:
            vel_data_numpy = np.zeros_like(data_numpy)
            vel_data_numpy[:-1,:] = data_numpy[1:,:] - data_numpy[:-1,:]
            vel_data_numpy[-1,:] = 0
            vel_data_numpy = vel_data_numpy
            
            #vel_data_numpy_masked = np.zeros_like(data_numpy_masked)
            #vel_data_numpy_masked[:-1,:] = data_numpy_masked[1:,:] - data_numpy_masked[:-1,:]
            #vel_data_numpy_masked[-1,:] = 0
            #vel_data_numpy = self.preprocess_norm(vel_data_numpy, length)
            
        #for func in self.transforms:

        #image = torch.Tensor(np.resize(image,(256,256,3)).astype(np.float64)).permute(2,0,1) #self.transforms(data_numpy)
        #bone_data_numpy = torch.Tensor(np.resize(bone_data_numpy,(256,256,3))).permute(2,0,1)#self.transforms(bone_data_numpy)
        #vel_data_numpy = torch.Tensor(np.resize(vel_data_numpy,(256,256,3))).permute(2,0,1) #self.transforms(vel_data_numpy)
        #return data_all
        #print(torch.Tensor(label))
        #data_numpy_masked = torch.Tensor(np.resize(data_numpy_masked,(256,256,3)).astype(np.float64)).permute(2,0,1) #self.transforms(data_numpy)

        #bone_data_numpy_masked = torch.Tensor(np.resize(bone_data_numpy_masked,(256,256,3))).permute(2,0,1)#self.transforms(bone_data_numpy)
        #vel_data_numpy_masked = torch.Tensor(np.resize(vel_data_numpy_masked,(256,256,3))).permute(2,0,1) #self.transforms(vel_data_numpy) 
        #data_numpy = self.resize(data_numpy.astype(np.uint8))
        #bone_data_numpy = self.resize(bone_data_numpy.astype(np.uint8))
        #vel_data_numpy = self.resize(vel_data_numpy.astype(np.uint8))
        data_numpy = np.concatenate([data_numpy, data_numpy[:,:,:1]], axis=-1)
        #bone_data_numpy = torch.cat([bone_data_numpy,bone_data_numpy[:1]], dim=0)
        #vel_data_numpy = torch.cat([vel_data_numpy, vel_data_numpy[:1]], dim=0)
        t,n,c = data_numpy.shape
        if t%3 !=0:
            data_numpy =  np.concatenate([data_numpy, np.zeros([3-t%3, n, c])], axis=0)
        k = data_numpy.shape[0]
        data_numpy = np.transpose(data_numpy.reshape(3,k//3, n,c), [0,2,1,3]).reshape(3,n,k*c//3)
        data_numpy = np.transpose(data_numpy, [1,2,0])  
        data_numpy = torch.Tensor(np.resize(data_numpy,(256,256,3)).astype(np.float64)).permute(2,0,1) #self.transforms(data_numpy)
        #bone_data_numpy = torch.Tensor(np.resize(bone_data_numpy,(256,256,3)).astype(np.float64)).permute(2,0,1)#self.transforms(bone_data_numpy)
        #vel_data_numpy = torch.Tensor(np.resize(vel_data_numpy,(256,256,3)).astype(np.float64)).permute(2,0,1) #self.transforms(vel_data_numpy)    
        return  data_numpy, torch.Tensor([label]).long() #torch.stack([data_numpy, bone_data_numpy, vel_data_numpy], dim=0), torch.Tensor([label]).long()

#data_numpy, torch.Tensor([label]).long()