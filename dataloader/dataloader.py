import numpy as np

from torch.utils.data import Dataset

import tools_dataset
import pickle as pkl
import torch
ntu_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)
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
        self.bone = bone
        self.vel = vel
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
        '''
        x_min = np.min(data[...,0])
        x_max = np.max(data[...,0])
        y_min = np.min(data[...,1])
        y_max = np.max(data[...,1])
        z_min = np.min(data[...,2])
        z_max = np.max(data[...,2])

        if x_max != x_min:
            data[...,0] = (data[...,0] - x_min)*255/(x_max - x_min)
        else:
            data[...,0] = np.zeros_like(data[...,0])
        if y_max != y_min:
            data[...,1] = (data[...,1] - y_min)*255/(y_max - y_min)
        else:
            data[...,1] = np.zeros_like(data[...,0])
        if z_max != z_min:
            data[...,2] = (data[...,2] - z_min)*255/(z_max - z_min)
        else:
            data[...,2] = np.zeros_like(data[...,0])
        '''
        return data
    def __getitem__(self, ind):
        index = self.index_list[ind]
        self.bone=True
        self.vel=True
        data_numpy = self.data_info[index]
        length = self.length_info[index]
        #print(ind)
        label = self.targets[ind]
        key = self.key_info[index]
        name = key.split('/')[-1].split('.')[0]
        #path_3d = '/export/md0/datasets/NTU_RGBD/heatmap_joints/' + name + '.mp4'
        #video = extract_frames(path_3d)
        #data_numpy = np.array(data_numpy)
        #valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        #data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        #if self.random_rot:
        #    data_numpy = tools.random_rot(data_numpy)
        data_numpy = self.preprocess(data_numpy, length) # t,m,n,c
        if self.bone:
            #from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, v1 - 1] = data_numpy[:, v1 - 1] - data_numpy[:, v2 - 1]
            bone_data_numpy = self.preprocess_norm(bone_data_numpy, length)
        if self.vel:
            vel_data_numpy = np.zeros_like(data_numpy)
            vel_data_numpy[:-1,:] = data_numpy[1:,:] - data_numpy[:-1,:]
            vel_data_numpy[-1,:] = 0
            vel_data_numpy = self.preprocess_norm(vel_data_numpy, length)
        #for func in self.transforms:
        data_numpy = torch.Tensor(np.resize(data_numpy,(384,384,3)).astype(np.float64)).permute(2,0,1) #self.transforms(data_numpy)
        #bone_data_numpy = torch.Tensor(np.resize(bone_data_numpy,(256,256,3))).permute(2,0,1)#self.transforms(bone_data_numpy)
        #vel_data_numpy = torch.Tensor(np.resize(vel_data_numpy,(256,256,3))).permute(2,0,1) #self.transforms(vel_data_numpy)
        #return data_all
        #print(torch.Tensor(label))
        
        return data_numpy, torch.Tensor([label]).long() #, torch.stack([data_numpy, bone_data_numpy, vel_data_numpy], axis=0), torch.Tensor([label]).long()

#data_numpy, torch.Tensor([label]).long()