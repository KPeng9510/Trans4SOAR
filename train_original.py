# The testing module requires faiss
# So if you don't have that, then this import will break
from tkinter import Label
from pml_origin import losses, miners, samplers, trainers, testers
from utils import stat_utils
import torch.nn as nn
import record_keeper
import pml_origin.utils.logging_presets as logging_presets            
from torchvision import datasets, models, transforms
import torchvision
import logging
logging.getLogger().setLevel(logging.INFO)
import os
import numpy as np
from dataloader.dataloader_three import Feeder
#from loader_three_ntu60 import Feeder
import pml_origin as pytorch_metric_learning
from pml_origin.testers.base_tester import BaseTester
logging.info("pytorch-metric-learning VERSION %s"%pytorch_metric_learning.__version__)
logging.info("record_keeper VERSION %s"%record_keeper.__version__)
from models.Trans4SOAR import Trans4SOAR_base
from sklearn.metrics import accuracy_score
#from efficientnet_pytorch import EfficientNet
import torch
import sklearn
import hydra
from omegaconf import DictConfig
# reprodcibile
#seed=42
#np.random.seed(42)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(128, 256), nn.ReLU(),nn.BatchNorm1d(256), nn.Linear(256,20), nn.Softmax(dim=-1))        

    def forward(self, x):
        return self.fc1(x)
class OneShotTester(BaseTester):

    def __init__(self, end_of_testing_hook=None):
        super().__init__()
        self.max_accuracy = 0.0
        self.embedding_filename = ""
        self.end_of_testing_hook = end_of_testing_hook


    def __get_correct(self, output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
    #             print(correct)
        return correct


    def __accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            correct = self.__get_correct(output, target, topk)
            batch_size = target.size(0)
            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def do_knn_and_accuracies(self, accuracies, embeddings_and_labels, split_name, tag_suffix=''):
        query_embeddings = embeddings_and_labels["val"][0]
        query_labels = embeddings_and_labels["val"][1]
        reference_embeddings = embeddings_and_labels["samples"][0]
        reference_labels = embeddings_and_labels["samples"][1]
        knn_indices, knn_distances = stat_utils.get_knn(reference_embeddings.astype('float32'),
                                                              query_embeddings.astype('float32'), 1, False)
        knn_labels = reference_labels[knn_indices][:, 0]
        accuracy = accuracy_score(knn_labels, query_labels)
        f_1_score = sklearn.metrics.f1_score(query_labels, knn_labels, average='macro')
        precision = sklearn.metrics.precision_score(query_labels, knn_labels, average='macro')
        recall = sklearn.metrics.recall_score(query_labels, knn_labels, average='macro')
        logging.info('accuracy:{}'.format(accuracy))
        logging.info('f_1_score:{}'.format(f_1_score))
        logging.info('precision:{}'.format(precision))
        logging.info('recall:{}'.format(recall))
        keyname = self.accuracies_keyname("mean_average_precision_at_r")  # accuracy as keyname not working
        accuracies[keyname] = accuracy
class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)
class MLP_Drop(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Linear(input_size, curr_size))
        layer_list.append(torch.nn.Dropout(0.5))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)

# This is for replacing the last layer of a pretrained network.
# This code is from https://github.com/KevinMusgrave/powerful_benchmarker
class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def get_datasets(data_dir, cfg, mode="train"):

    common_transforms = []
    train_transforms = []
    test_transforms = []
    #if cfg.transform.transform_resize_match:
    common_transforms.append(transforms.Resize((cfg.transform.transform_resize,cfg.transform.transform_resize)))
    
    if cfg.transform.transform_random_resized_crop:
        train_transforms.append(transforms.RandomResizedCrop(cfg.transform.transform_resize))
    if cfg.transform.transform_random_horizontal_flip:
        train_transforms.append(torchvision.transforms.RandomHorizontalFlip(p=0.5))
    if cfg.transform.transform_random_rotation:
        train_transforms.append(transforms.RandomRotation(cfg.transform.transform_random_rotation_degrees))#, fill=255))
    if cfg.transform.transform_random_shear:
        train_transforms.append(torchvision.transforms.RandomAffine(0,
                                                                    shear=(
                                                                        cfg.transform.transform_random_shear_x1,
                                                                        cfg.transform.transform_random_shear_x2,
                                                                        cfg.transform.transform_random_shear_y1,
                                                                        cfg.transform.transform_random_shear_y2
                                                                        ),
                                                                    fillcolor=255)) 
    if cfg.transform.transform_random_perspective:
        train_transforms.append(transforms.RandomPerspective(distortion_scale=cfg.transform.transform_perspective_scale, 
                                     p=0.5, 
                                     interpolation=3)
                                )
    if cfg.transform.transform_random_affine:
        train_transforms.append(transforms.RandomAffine(degrees=(cfg.transform.transform_degrees_min,
                                                                 cfg.transform.transform_degrees_max),
                                                        translate=(cfg.transform.transform_translate_a,
                                                                   cfg.transform.transform_translate_b),
                                                        fillcolor=255))
    data_transforms = {
            'train': transforms.Compose(common_transforms+train_transforms+[transforms.ToTensor()]),
            'test': transforms.Compose(common_transforms+[transforms.ToTensor()]),
            }
    train_dataset = Feeder(mode='train', transforms=data_transforms["train"])





    # for the final model we can join train, validation, validation samples datasets
    print(mode)
    if mode == "final_train":

        test_dataset = Feeder(mode='test', transforms=data_transforms["test"])
        samples_dataset = Feeder(mode='val',transforms=data_transforms["test"])

        return train_dataset, test_dataset, samples_dataset#,samples_dataset2,samples_dataset3,samples_dataset4,samples_dataset5, samples_dataset6
    else:#
        if mode == "train":
            val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"),
                    data_transforms["test"])

            val_samples_dataset = datasets.ImageFolder(os.path.join(data_dir, "val_samples"),
                    data_transforms["test"])
            return train_dataset, val_dataset, val_samples_dataset

        if mode == "test":
            return train_dataset, test_dataset, samples_dataset


@hydra.main(config_path="config/config.yaml")
def train_app(cfg):
    print(cfg.pretty())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trunk = Trans4SOAR_base(pretrained=False)#poolformer_s12(pretrained=False)
    embedder = MLP([512, cfg.embedder.size])
    classifier = MLP([cfg.embedder.size, 100])
    trunk = torch.nn.DataParallel(trunk.to(device))
    embedder = torch.nn.DataParallel((embedder).to(device))
    classifier = torch.nn.DataParallel(classifier.to(device))

    # Set optimizers
    if cfg.optimizer.name == "sdg":
        trunk_optimizer = torch.optim.SGD(trunk.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
        embedder_optimizer = torch.optim.SGD(embedder.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
        classifier_optimizer = torch.optim.SGD(classifier.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
    elif cfg.optimizer.name == "rmsprop":
        
        #trunk_optimizer = torch.optim.RMSprop(trunk.parameters(), lr=0.000035, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
        #embedder_optimizer = torch.optim.RMSprop(embedder.parameters(), lr=0.000035, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
        #classifier_optimizer = torch.optim.RMSprop(classifier.parameters(), lr=0.000035, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
        
        trunk_optimizer = torch.optim.AdamW(trunk.parameters(), lr=0.000035, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        embedder_optimizer = torch.optim.AdamW(embedder.parameters(), lr=0.000035, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        classifier_optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.000035, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    else:
        trunk_optimizer = torch.optim.AdamW(trunk.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False,  maximize=False)
        embedder_optimizer = torch.optim.AdamW(embedder.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False,  maximize=False)
        classifier_optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False,  maximize=False)


    # Set the datasets
    data_dir = os.environ["DATASET_FOLDER"]+"/"+cfg.dataset.data_dir
    print("Data dir: "+data_dir)

    train_dataset, val_dataset, val_samples_dataset = get_datasets(data_dir, cfg, mode=cfg.mode.type) #,samples_dataset2,samples_dataset3,samples_dataset4,samples_dataset5,samples_dataset6
    print("Trainset: ",len(train_dataset), "Testset: ",len(val_dataset), "Samplesset: ",len(val_samples_dataset))

    # Set the loss function
    if cfg.embedder_loss.name == "margin_loss":
        loss = losses.MarginLoss(margin=cfg.embedder_loss.margin,nu=cfg.embedder_loss.nu,beta=cfg.embedder_loss.beta)
    if cfg.embedder_loss.name == "triplet_margin":
        loss = losses.TripletMarginLoss(margin=cfg.embedder_loss.margin)
    if cfg.embedder_loss.name == "multi_similarity":
        loss = losses.MultiSimilarityLoss(alpha=cfg.embedder_loss.alpha, beta=cfg.embedder_loss.beta, base=cfg.embedder_loss.base)

    # Set the classification loss:
    classification_loss = torch.nn.CrossEntropyLoss()

    # Set the mining function

    if cfg.miner.name == "triplet_margin":
        #miner = miners.TripletMarginMiner(margin=0.2)
        miner = miners.TripletMarginMiner(margin=cfg.miner.margin)
    if cfg.miner.name == "multi_similarity":
        miner = miners.MultiSimilarityMiner(epsilon=cfg.miner.epsilon)
        #miner = miners.MultiSimilarityMiner(epsilon=0.05)

    batch_size = 32 #cfg.trainer.batch_size
    num_epochs = cfg.trainer.num_epochs
    iterations_per_epoch = cfg.trainer.iterations_per_epoch
    # Set the dataloader sampler
    sampler = samplers.MPerClassSampler(train_dataset.targets, m=4, length_before_new_iter=len(train_dataset))


    # Package the above stuff into dictionaries.
    models = {"trunk": trunk, "embedder": embedder, "classifier": classifier}
    optimizers = {"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer, "classifier_optimizer": classifier_optimizer}
    loss_funcs = {"metric_loss": loss, "classifier_loss": classification_loss}
    mining_funcs = {"tuple_miner": miner}

    # We can specify loss weights if we want to. This is optional
    loss_weights = {"metric_loss": cfg.loss.metric_loss, "classifier_loss": cfg.loss.classifier_loss}
    
    T_0 = 200*2962 #250 200 300 2962
    
    schedulers = {
            #"metric_loss_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(classifier_optimizer, cfg.scheduler.step_size, gamma=cfg.scheduler.gamma),
            "embedder_scheduler_by_epoch": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(embedder_optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False),
            "classifier_scheduler_by_epoch": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(classifier_optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False),
            "trunk_scheduler_by_epoch": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(trunk_optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False),
            }
    '''
    
    
    schedulers = {
            #"metric_loss_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(classifier_optimizer, cfg.scheduler.step_size, gamma=cfg.scheduler.gamma),
            "embedder_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(embedder_optimizer, 10, gamma=0.1),
            "classifier_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(classifier_optimizer, 10, gamma=0.1),
            "trunk_scheduler_by_epoch": torch.optim.lr_scheduler.StepLR(trunk_optimizer, 10, gamma=0.1),
            } # cfg.scheduler.step_size
    '''
    experiment_name = "%s_model_%s_cl_%s_ml_%s_miner_%s_mix_ml_%02.2f_mix_cl_%02.2f_resize_%d_emb_size_%d_class_size_%d_opt_%s_lr_%02.2f_m_%02.2f_wd_%02.2f"%(cfg.dataset.name,
                                                                                                  cfg.model.model_name, 
                                                                                                  "cross_entropy", 
                                                                                                  cfg.embedder_loss.name, 
                                                                                                  cfg.miner.name, 
                                                                                                  cfg.loss.metric_loss, 
                                                                                                  cfg.loss.classifier_loss,
                                                                                                  cfg.transform.transform_resize,
                                                                                                  cfg.embedder.size,
                                                                                                  cfg.embedder.class_out_size,
                                                                                                  cfg.optimizer.name,
                                                                                                  cfg.optimizer.lr,
                                                                                                  cfg.optimizer.momentum,
                                                                                                  cfg.optimizer.weight_decay)
    record_keeper, _, _ = logging_presets.get_record_keeper("logs/%s"%(experiment_name), "tensorboard/%s"%(experiment_name))
    hooks = logging_presets.get_hook_container(record_keeper)
    dataset_dict = {"samples": val_samples_dataset, "val": val_dataset} #, 'train':train_dataset
    model_folder = "experimental_results/%sp/"%(experiment_name)

    # Create the tester
    tester = OneShotTester(
            end_of_testing_hook=hooks.end_of_testing_hook, 
            #size_of_tsne=20
            )
    #tester.embedding_filename=data_dir+"/embeddings_pretrained_triplet_loss_multi_similarity_miner.pkl"
    tester.embedding_filename=data_dir+"/"+experiment_name+".pkl"
    end_of_epoch_hook = hooks.end_of_epoch_hook(tester, dataset_dict, model_folder)
    trainer = trainers.TrainWithClassifier(models,
            optimizers,
            batch_size,
            loss_funcs,
            mining_funcs,
            train_dataset,
            sampler=sampler,
            lr_schedulers=schedulers,
            dataloader_num_workers = cfg.trainer.batch_size,
            loss_weights=loss_weights,
            end_of_iteration_hook=hooks.end_of_iteration_hook,
            end_of_epoch_hook=end_of_epoch_hook
            )

    trainer.train(num_epochs=num_epochs)

    tester = OneShotTester()

if __name__ == "__main__":
    train_app()
