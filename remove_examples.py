import os, sys
from utils import *
from models import *
from dataloader import *
import pickle
import params, json

def get_ids(num_ex, metric, acc_pre, acc_ft):
    learning_epoch = get_first_epoch_where_we_learn_forever(acc_pre)
    cum_learning_acc = acc_pre.sum(dim = 0)
    forgetting_epoch = get_first_epoch_where_we_forget_forever(acc_ft)
    cum_forgetting_acc = acc_ft.sum(dim = 0)

    from scipy.stats import rankdata
    ranks_learn = rankdata(learning_epoch, method='min') - 1
    ranks_learn = ranks_learn.max() - ranks_learn
    ranks_forget = rankdata(forgetting_epoch, method='min') - 1
    ranks_learn_cum = rankdata(cum_learning_acc, method='min') - 1
    ranks_forget_cum = rankdata(cum_forgetting_acc, method='min') - 1
    ranks_joint = ranks_learn_cum + ranks_forget_cum*100

    total_ex = ranks_joint.shape[0]
    all_ids = np.arange(total_ex)

    if metric == "random":
        #remove random
        idx = np.random.choice(all_ids, size=num_ex, replace=False, p=None)
    else:    
        rank_mapper =  {"cum_learn":ranks_learn_cum,
                        "cum_forget":ranks_forget_cum,
                        "forget_time":ranks_forget,
                        "learn_time":ranks_learn,
                        "joint":ranks_joint,
                        }
        rank = rank_mapper[metric]
        arr1inds = rank.argsort()
        sorted_idx = all_ids[arr1inds]
        idx = sorted_idx[:num_ex]

    mask = torch.zeros(total_ex)
    mask[idx] = 1
    return mask
    

## Accuracy after training with samples removed
def create_data_removed_loader(pre_dict, num_ex, metric, acc_pre, acc_ft):
    if num_ex == 0: return pre_dict["train_loader"]
    mask = get_ids(num_ex, metric, acc_pre, acc_ft)
    dataset = copy.deepcopy(pre_dict["train_dataset"])
    if type(dataset.targets == type([])):  dataset.targets = torch.tensor(dataset.targets)
    dataset.targets = dataset.targets[mask==0]
    dataset.data = dataset.data[mask==0]
    print("New Dataset Size =", dataset.data.shape[0])
    new_loader = DataLoader(dataset=dataset,batch_size=512, shuffle=True, num_workers=16)
    return new_loader

def trainer(all_args):
    #iid setting
    all_args["noise_2"] = all_args["noise_1"]
    if all_args["dataset1"] in ["cifar-5m","cifar10_500k","cifar10_dcgan"]:
        all_args["dataset2"] = "cifar10"
        #no need to split the dataset
        split_dataset = False
    else:
        all_args["dataset2"] = all_args["dataset1"]
        #split the dataset to make two splits
        split_dataset = True

    all_args["minority_2"] = all_args["minority_1"]

    pre_dict, ft_dict = return_loaders(all_args, get_frac = split_dataset)
    
    num_examples_removed = [0,50, 100, 200, 400, 800, 1600, 3200, 6400,12800]

    #load pre and ft masks
    filename = f'masks/{all_args["dataset1"]}/{all_args["lr1"]}_{all_args["lr2"]}_{all_args["noise_1"]}_{all_args["model_type"]}_{all_args["sched"]}_0'
    with open(f'{filename}_learn.pickle', 'rb') as handle:
        pre = pickle.load(handle)

    with open(f'{filename}_forget.pickle', 'rb') as handle:
        ft = pickle.load(handle)
    acc_pre = pre["acc_mask"]
    acc_ft = ft["acc_mask"]

    torch.manual_seed(100)
    n_repeats = 5
    for nr in range(n_repeats):
        for num_ex in num_examples_removed:
            in_channels = 1 if all_args["dataset1"] in ["mnist","fashionmnsit","emnist"] else 3
            model = get_model(all_args["model_type"], in_channels=in_channels, NUM_CLASSES=10)
            loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0)

            filename = f'example_removal/{all_args["dataset1"]}/{all_args["removal_metric"]}_{all_args["lr1"]}_{all_args["lr2"]}_{all_args["noise_1"]}_{all_args["model_type"]}_{all_args["sched"]}_{num_ex}_{nr}'
            print("Location:", filename)
            # we load seed after getting the dataset. 
            # Dataset indices are always chosen with the same fixed seed
            #STAGE 1
            
            
            loader = create_data_removed_loader(pre_dict, num_ex, all_args["removal_metric"], acc_pre, acc_ft)

            optimizer = SGD(model.parameters(), lr=all_args["lr1"], momentum=0.9, weight_decay=all_args["wd"])
            scheduler, EPOCHS = get_scheduler_epochs(all_args["sched"], optimizer, loader)

            mask_ret = train(model, loader, optimizer, scheduler, 
                                    loss_fn, EPOCHS = EPOCHS, eval_every = False, 
                                    eval_loader= None)

            #Evaluate
            #using ft_dict because it is some as pre_dict test loader everywhere except cifar-5m
            eval_ret = eval(model, ft_dict["test_loader"], eval_mode = True)
            with open(f'{filename}_eval.pickle', 'wb') as handle:
                pickle.dump(eval_ret, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = params.parse_args()
    args = parser.parse_args()
    args = params.add_config(args) if args.config_file != None else args
    args = vars(args)
    print(args)
    trainer(args)