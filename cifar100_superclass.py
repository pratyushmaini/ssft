import os, sys, random
from utils import *
from models import *
from dataloader import *

from torch.optim import SGD
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0)

def get_masks(model, trainloader,lr,wd,schedule, eval_loader):
    loader = trainloader
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler, EPOCHS = get_scheduler_epochs(schedule, optimizer, loader)
    masks = train(model, loader, optimizer, scheduler, loss_fn, EPOCHS = EPOCHS,
        eval_every = True, eval_loader= eval_loader)
    return masks


def get_accuracy_per_subgroup(mask, dset):
    groups = dset.groups
    num_groups=100
    accs = np.zeros(num_groups)
    nums = np.zeros(num_groups)
    for label in range(num_groups):
        nums[label] = (groups==label).sum()
        accs[label] = mask[groups == label].sum()
    
    accs = accs/nums
    return accs, nums

def get_nums_per_subgroup(dset):
    groups = dset.groups
    num_groups=100
    nums = np.zeros(num_groups)
    for group in range(num_groups):
        nums[group] = (groups==group).sum()
    
    return nums


def get_acc_list(all_args, n_train_repeats = 10, iter=0):
    pre_dict, ft_dict = return_loaders(all_args)
    
    learn_masks = []
    forget_masks = []
    for i in range(n_train_repeats):
        model = ResNet9(20).to(memory_format=torch.channels_last).cuda()
        #Learning
        preloader = pre_dict["train_loader"]
        new_mask = get_masks(model, preloader, all_args["lr1"], all_args["wd"], all_args["schedule"], preloader )
        num_rows = new_mask.shape[0]
        reqd_rows = 100 #num_epochs
        if num_rows < reqd_rows:
            new_arr = torch.ones(reqd_rows - num_rows, new_mask.shape[1])
            new_mask = torch.cat([new_mask, new_arr])
        learn_masks.append(new_mask)

        #Forgetting 
        ftloader = ft_dict["train_loader"]
        new_mask = get_masks(model, ftloader, all_args["lr2"], all_args["wd"], all_args["schedule"], preloader)
        num_rows = new_mask.shape[0]
        reqd_rows = 100 #num_epochs
        if num_rows < reqd_rows:
            new_arr = torch.zeros(reqd_rows - num_rows, new_mask.shape[1])
            new_mask = torch.cat([new_mask, new_arr])
        forget_masks.append(new_mask)

        torch.save(learn_masks[-1], f"masks/cifar100/{log}/learn_{i}_{iter}_lr_{peak_lr_ft}.pt")
        torch.save(forget_masks[-1], f"masks/cifar100/{log}/forget_{i}_{iter}_lr_{peak_lr_ft}.pt")

    noise_mask= pre_dict["noise_mask"]
    nums = get_nums_per_subgroup(pre_dict["train_dataset"])
    color = nums[pre_dict["train_dataset"].groups]
    torch.save(torch.from_numpy(color), f"masks/cifar100/{log}/color_{iter}_lr_{peak_lr_ft}.pt")
    torch.save(torch.from_numpy(noise_mask), f"masks/cifar100/{log}/noise_{iter}_lr_{peak_lr_ft}.pt")

    # acc_list = []
    # for mask in learn_masks:
    #     acc, nums = get_accuracy_per_subgroup(mask, pre_dict["train_dataset"])
    #     acc_list.append(torch.from_numpy(acc).unsqueeze(0))
    # acc_list = torch.cat(acc_list).numpy()
    # idx = np.argsort(nums)[::-1]
    # acc_list = acc_list[:,idx]
    return None


dataset1 = "cifar100"
dataset2 = "cifar100"
weight_decay = 5e-4
peak_lr_pre = 0.1

label_noise_ratio_pre = 0.1
label_noise_ratio_ft = 0.1
model_type = "resnet-9" #or lenet
schedule = "triangle" #or step
seed = 0
minority_1 = 0
minority_2 = 0
batch_size = 512


peak_lr_ft = 0.05
log_factor = 2
log = f"LOG{log_factor}"

all_args = {"dataset1":dataset1, "dataset2":dataset2, "wd":weight_decay, "lr1":peak_lr_pre, "lr2":peak_lr_ft, 
            "noise_1":label_noise_ratio_pre, "noise_2":label_noise_ratio_ft, "model":model_type, "schedule":schedule,
            "minority_1":minority_1, "minority_2":minority_2, "seed":seed, "batch_size":batch_size, "log_factor":log_factor}


n_sample_repeats = 10
n_trial_repeats = 5
acc_list = None
for i in range(n_sample_repeats):
    print("Exp Runs Completed = ", i)
    get_acc_list(all_args, n_trial_repeats, i) 

# acc_list = acc_list/n_sample_repeats
# torch.save(torch.from_numpy(acc_list), "data/cifar100_acc_lists_{log}_forget_single.pt")