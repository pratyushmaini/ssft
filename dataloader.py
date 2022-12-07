from torchvision import transforms,datasets
import torch
import numpy as np
from torch.utils.data import DataLoader
import random, copy
import os
from utils import *



def seed_everything(seed: int):
    # print("setting seed", seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

#cifar100 subgroup labels
coarse_labels = {   0: [72, 4, 95, 30, 55],
                    1: [73, 32, 67, 91, 1],
                    2: [92, 70, 82, 54, 62],
                    3: [16, 61, 9, 10, 28],
                    4: [51, 0, 53, 57, 83],
                    5: [40, 39, 22, 87, 86],
                    6: [20, 25, 94, 84, 5],
                    7: [14, 24, 6, 7, 18],
                    8: [43, 97, 42, 3, 88],
                    9: [37, 17, 76, 12, 68],
                    10: [49, 33, 71, 23, 60],
                    11: [15, 21, 19, 31, 38],
                    12: [75, 63, 66, 64, 34],
                    13: [77, 26, 45, 99, 79],
                    14: [11, 2, 35, 46, 98],
                    15: [29, 93, 27, 78, 44],
                    16: [65, 50, 74, 36, 80],
                    17: [56, 52, 47, 59, 96],
                    18: [8, 58, 90, 13, 48],
                    19: [81, 69, 41, 89, 85]
                }


class TensorDataset(torch.utils.data.Dataset):
    # use for imagenette, cifar-dcgan, cifar-5m
    def __init__(self, data_path, split):
        root = data_path
        self.split = split
        self.data = torch.load(f"{root}/{split}_x.pt")
        self.targets = torch.load(f"{root}/{split}_y.pt").long()
        self.n_classes = torch.unique(self.targets).shape[0]
        self.transform = None

    def __getitem__(self, index):
        x_data_index = self.data[index]
        if self.transform:
            x_data_index = self.transform(x_data_index)
        return (x_data_index, self.targets[index], index)

    def __len__(self):
        return self.data.shape[0]


def get_cifar_2(root = './data', download=False, train=True):
    tvs = transforms.ToTensor()
    cifar = dataset_with_indices(datasets.CIFAR10)(root=root,download=download,train=train,transform=tvs)
    d1 = copy.deepcopy(cifar)
    new_cifar_data = cifar.data
    new_cifar_targets = torch.tensor(cifar.targets)
    #plane and horse
    ids = ((new_cifar_targets==0) + (new_cifar_targets==7))
    new_cifar_data = new_cifar_data[ids]
    new_cifar_targets = new_cifar_targets[ids]
    new_cifar_targets[new_cifar_targets==7] = 1
    d1.data = new_cifar_data.astype('uint8')
    d1.targets = new_cifar_targets
    return d1

def get_mnist_cifar_union(root = './data', download=False, train=True):

    tvs = transforms.ToTensor()
    mnist = dataset_with_indices(datasets.MNIST)(root=root,download=download,train=train,transform=tvs)
    cifar = dataset_with_indices(datasets.CIFAR10)(root=root,download=download,train=train,transform=tvs)
    
    d1 = copy.deepcopy(cifar)
    #mnist modify
    new_mnist_data = mnist.data.unsqueeze(1).repeat(1,3,1,1)
    new_mnist_data = torch.nn.functional.pad(input=new_mnist_data, pad=(2, 2, 2, 2), mode='constant', value=0)
    # new_mnist_data = ((new_mnist_data/255.-MNIST_MEAN)/MNIST_STD)
    # new_mnist_data *= 255
    new_mnist_targets = mnist.targets
    ids = (new_mnist_targets==7) + (new_mnist_targets==4) + (new_mnist_targets==5) + (new_mnist_targets==6)
    new_mnist_data = new_mnist_data[ids]
    new_mnist_targets = new_mnist_targets[ids]
    new_mnist_targets[new_mnist_targets==7] = 1
    new_mnist_targets[new_mnist_targets==4] = 0
    new_mnist_targets[new_mnist_targets==5] = 2
    new_mnist_targets[new_mnist_targets==6] = 3
    

    #cifar10 modify
    new_cifar_data = torch.from_numpy(cifar.data).permute(0,3,1,2)
    # new_cifar_data = (new_cifar_data/255. - get_tensorized(CIFAR_MEAN))/get_tensorized(CIFAR_STD)
    # new_cifar_data *= 255
    new_cifar_targets = torch.tensor(cifar.targets)
    #plane and horse
    ids = ((new_cifar_targets==0) + (new_cifar_targets==7)) + (new_cifar_targets==5) + (new_cifar_targets==6)
    new_cifar_data = new_cifar_data[ids]
    new_cifar_targets = new_cifar_targets[ids]
    new_cifar_targets[new_cifar_targets==7] = 1
    new_cifar_targets[new_cifar_targets==5] = 2
    new_cifar_targets[new_cifar_targets==6] = 3

    print(new_mnist_targets.shape, new_cifar_targets.shape)

    d1.data = torch.cat([new_mnist_data, new_cifar_data]).permute(0,2,3,1).numpy().astype('uint8')
    d1.targets = torch.cat([new_mnist_targets, new_cifar_targets])
    print(d1.targets.shape)
    return d1

def get_cifar100_superclass(root, download=False, train=True, log_factor = 2, seed = 0):
    n_classes = 20
    tvs = transforms.Compose([transforms.ToTensor()])
    # tvs = None
    d_func = datasets.CIFAR100
    dset = dataset_with_indices(d_func)(root, download=download, train=train, transform=tvs)    
    # dset = d_func(root, download=download, train=train, transform=tvs)    
    dset.targets = torch.tensor(dset.targets)
    #for each class in the superclasses reduce sample size by log_factor
    dset.group_counts = torch.zeros(dset.targets.shape[0])
    if train:
        for i in range(n_classes): #20
            num_per_class = 500 #default
            #for each super class do the following
            #shuffle the order in which the sub labels appear
            sub_labels = copy.deepcopy(coarse_labels[i])
            random.shuffle(sub_labels)
            # print(i, sub_labels)
            for iter, idx in enumerate(sub_labels):
                #for each class in the super class do the following
                mask = torch.ones(dset.targets.shape[0]) #need to check current size as this is changing
                ns = int(num_per_class/(log_factor**iter))
                #create a new mask that only selects with ns samples from num_labels
                new_mask = torch.zeros(num_per_class)
                new_mask[:ns] = 1
                #randomly permute this so we dont always select the first k training samples
                perm = torch.randperm(new_mask.shape[0])
                new_mask = new_mask[perm]
                mask[dset.targets == idx] = new_mask
                dset.data = dset.data[mask == 1]
                dset.targets = dset.targets[mask == 1]
                dset.group_counts =  dset.group_counts[mask==1]
                dset.group_counts[dset.targets == idx] = ns
            
    #set the super class label. doing separately to avoid overwriting issues
    dset.new_targets = dset.targets.clone()
    dset.groups = dset.targets.clone()
    
    for i in range(20):       
        for idx in coarse_labels[i]:
            dset.new_targets[dset.targets == idx] = i

    dset.targets = dset.new_targets
    #sort the dataset based on the group ids so that the random noise masking can happen at the same relative
    arr1inds = dset.group_counts.argsort()
    dset.data = dset.data[arr1inds]
    dset.targets = dset.targets[arr1inds]
    dset.groups = dset.groups[arr1inds]
    dset.group_counts = dset.group_counts[arr1inds]

    return n_classes, dset

def get_split_ids(dataset_size, ratio):
    indices = list(range(dataset_size))
    random.Random(0).shuffle(indices)
    split = int(dataset_size*ratio)
    pre_indices, ft_indices = indices[split:], indices[:split]
    pre_indices.sort()
    ft_indices.sort()
    # ipdb.set_trace()
    return pre_indices, ft_indices

def corrupt_labels(dset, n_classes, corrupt_prob, seed = 0, label_noise = True):
    labels = np.array(dset.targets)
    #Intialise a random number generator
    rng = np.random.default_rng(seed)
    # mask = rng.random(len(labels)) <= corrupt_prob
    num_examples = int(corrupt_prob*len(labels))
    idx = rng.choice(np.arange(len(labels)), num_examples, replace = False)
    mask = np.zeros(len(labels)).astype('int64')
    mask[idx] = 1
    if label_noise:
        #Random label should not coincide with true label
        if n_classes != 2: rnd_labels = rng.choice(n_classes - 2, num_examples) + 1 #we will do [(true + rand) % num_classes]
        else: rnd_labels = 1
        labels[idx] = (labels[idx] + rnd_labels) % n_classes
    else:
        rnd_labels = rng.choice(n_classes, num_examples)
        labels[idx] = rnd_labels
    labels = [int(x) for x in labels]
    dset.targets = labels
    return dset, mask

call_dataset = {"mnist":datasets.MNIST, 

                "cifar10":datasets.CIFAR10, 
                "emnist":datasets.EMNIST}


def make_grayscale(dset, ratio, seed  = 0):
    tv = torchvision.transforms.Grayscale(num_output_channels=1)
    total_ex = dset.data.shape[0]
    rng = np.random.default_rng(seed+1)
    num_examples = int(ratio*total_ex)
    idx = rng.choice(np.arange(total_ex), num_examples, replace = False)
    mask = np.zeros(total_ex).astype('int64')
    mask[idx] = 1
    dset.data[mask] = tv(torch.from_numpy(dset.data[mask]).permute(0,3,1,2)).permute(0,2,3,1)
    return dset, mask

def add_rare(original_dataset, dset, ratio, seed = 0):
    if original_dataset == "mnist":
        dataset = "emnist"
    elif original_dataset == "cifar10":
        return make_grayscale(dset, ratio, seed = 0)
    else:
        raise("not implemented")

    n_classes, new_dataset = return_basic_dset(dataset, "tr")
    num_samples =  int(ratio*dset.data.shape[0])
    dset.data = torch.cat([dset.data, new_dataset.data[:num_samples]])
    dset.targets = torch.cat([torch.tensor(dset.targets), new_dataset.targets[:num_samples]])
    return dset, None

def return_basic_dset(dataset, split, log_factor=2, seed_superclass = -1):
    train = True if split == "tr" else False
    if dataset == "cifar100" and seed_superclass != -1:
        #This is the rare example experiment
        seed_everything(seed_superclass)
        return get_cifar100_superclass("../data", download=True, train = train, log_factor=log_factor, seed=seed_superclass)
    
    
    tvs = transforms.Compose([transforms.ToTensor()])
    if dataset == "emnist":
        dset = dataset_with_indices(call_dataset[dataset])('./data', download=True, train=train, split='letters', transform=tvs)
        dset.data = dset.data[dset.targets < 10]
        dset.targets = dset.targets[dset.targets < 10]
    elif dataset == "mnist_cifar_union":
        dset = get_mnist_cifar_union('./data', download=True, train=train)
    elif dataset == "cifar2":
        dset = get_cifar_2('./data', download=True, train=train)
    elif dataset in ["cifar-5m", "cifar10_dcgan", "imagenette"]:
        dset = TensorDataset(f"./data/{dataset}", split)
        n_classes = dset.n_classes
    else:
        dset = dataset_with_indices(call_dataset[dataset])('./data', download=True, train=train, transform=tvs)
    
    try:
        n_classes = torch.tensor(dset.targets).max().item() + 1
    except:
        n_classes = dset.targets.max().item() + 1

    return n_classes, dset

def get_dset(split, dataset, noise_ratio, indices, minority_ratio = 0, seed = 0, log_factor = 2, seed_superclass=1):
    n_classes, dset = return_basic_dset(dataset, split, log_factor, seed_superclass)

    #get the correct slice
    if indices is not None:
        split_ratio = 0.5
        pre_indices, ft_indices =  get_split_ids(dset.data.shape[0], ratio = split_ratio)
        # print("Num indices less than 23446 = ", (torch.tensor(pre_indices) < 23446).sum().item())
        indices = pre_indices if indices == "pre" else ft_indices
        dset.data = dset.data[indices]
        try: dset.targets = dset.targets[indices]
        except: dset.targets = torch.tensor(dset.targets)[indices] #for cifar10
        if 'groups' in dset.__dict__:
            dset.groups = dset.groups[indices] #for cifar100 superclass
            dset.group_counts = dset.group_counts[indices] #for cifar100 superclass
            #but the group counts will also change!! because we removed many samples
            num_groups = dset.groups.max().item() + 1
            counts = torch.zeros(num_groups)  
            for g in range(num_groups):
                counts[g] = (dset.groups == g).sum()
                dset.group_counts[dset.groups == g] = counts[g]
                

    mask, mask2 = None, None
    if noise_ratio > 0: dset, mask = corrupt_labels(dset, n_classes, noise_ratio, seed)
    if minority_ratio > 0: 
        dset, mask2 = add_rare(dataset, dset, minority_ratio, seed)
    return dset, mask, mask2

def return_loaders(all_args, get_frac = True):
    split = "tr"
    indices1, indices2 = ("pre", "ft") if get_frac else (None, None)
    d1_tr, mask_noise1, mask_rare1 = get_dset(split, all_args["dataset1"], all_args["noise_1"], indices1, all_args["minority_1"], all_args["seed"], all_args["log_factor"], all_args["seed_superclass"])
    d2_tr, mask_noise2, mask_rare2 = get_dset(split, all_args["dataset2"], all_args["noise_2"], indices2 , all_args["minority_2"], all_args["seed"], all_args["log_factor"], all_args["seed_superclass"])
    batch_size = all_args["batch_size"]

    preloader = DataLoader(dataset=d1_tr, batch_size=batch_size, shuffle=True, num_workers=16)
    ftloader = DataLoader(dataset=d2_tr, batch_size=batch_size, shuffle=True, num_workers=16)

    #get test datasets
    split  = "te"
    d1, _, _ = get_dset(split, all_args["dataset1"], 0, None)
    d2, _, _ = get_dset(split, all_args["dataset2"], 0, None)

    preloader_test = DataLoader(dataset=d1, batch_size=batch_size, shuffle=False, num_workers=16)
    ftloader_test = DataLoader(dataset=d2, batch_size=batch_size, shuffle=False, num_workers=16)

    pre_dict = { "train_loader":preloader, 
                "test_loader":preloader_test,
                "noise_mask":mask_noise1,
                "rare_mask":mask_rare1,
                "train_dataset": d1_tr
              }

    ft_dict = { "train_loader":ftloader, 
                "test_loader":ftloader_test,
                "noise_mask":mask_noise2,
                "rare_mask":mask_rare2,
                "train_dataset": d2_tr
              }

    return pre_dict, ft_dict



def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    # def __init__()
    # self.indices = torch.arange(self.targets.shape[0])

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })
