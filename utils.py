import numpy as np
import torch
from matplotlib import pyplot as plt
import torchvision
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD, lr_scheduler

def get_model_id(args):
    ret = ""
    for name in ["model_type","noise_1","minority_1","lr2","sched","opt","batch_size","noise_2","minority_2","dataset2","wd","lr1"]:
        ret += name + "_" + str(args[name]) + "_"
    
    #remove last "_"
    ret = ret[:-1]
    return ret


CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

def get_tensorized(l):
    return torch.tensor(l).unsqueeze(-1).unsqueeze(-1)/255. 

def imshow(img, size = 4, loc = None):
    plt.rcParams['figure.figsize'] = [size, size]
    shape = img.shape
    dtype = type(img)
    #set dtype
    if dtype == type(np.zeros(1)):
        img = torch.from_numpy(img)
    
    #unsqueeze if mnist type
    if len(shape) == 3:
        img = img.unsqueeze(1)
    
    #permute if last dimension is 1 or 3
    if shape[-1] in [1,3]:
        img = img.permute(0,3,1,2)

    img = torchvision.utils.make_grid(img, nrow = size).cpu()
    npimg = img.numpy()
    plt.axis('off')
    
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if loc:
        plt.savefig(loc)


def get_scheduler_epochs(name, optimizer, loader, max_epochs = None):
    EPOCHS = max_epochs if max_epochs is not None else 100
    if name == "triangle":
        iters_per_epoch = len(loader)+1
        lr_schedule = np.interp(np.arange((EPOCHS+1) * iters_per_epoch),
                                [0, 10 * iters_per_epoch, EPOCHS * iters_per_epoch],
                                [0, 1, 0])
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
    elif name == "linear":
        iters_per_epoch = len(loader)+1
        lr_schedule = np.interp(np.arange((EPOCHS+1) * iters_per_epoch),
                                [0, EPOCHS * iters_per_epoch],
                                [0, 1])
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
    elif name == "cosine":
        iters_per_epoch = len(loader)+1
        T_max = EPOCHS*iters_per_epoch
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=- 1, verbose=False)
    else:
        assert(name == "step")
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, threshold=0.01, threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08, verbose=True)

    return scheduler, EPOCHS

    

def single_epoch(model, loader, opt, loss_fn, scheduler=None):
    model.train()
    scaler = GradScaler()
    total_correct = 0
    total_loss = 0
    total_num = 0
    mask_after_opt = torch.zeros(len(loader.dataset))
    conf_after_opt = torch.zeros(len(loader.dataset))

    for ims, labs, ids in loader:
        opt.zero_grad(set_to_none=True)
        ims, labs = ims.cuda(), labs.cuda()
        
        with autocast():
            out = model(ims)
            loss = loss_fn(out, labs)
            total_loss += loss.cpu().item()
            
            correct_mask = out.argmax(1).eq(labs)
            conf_mask = out[torch.arange(labs.shape[0]),labs]
            mask_after_opt[ids.squeeze(-1)] = correct_mask.float().cpu()
            conf_after_opt[ids.squeeze(-1)] = conf_mask.float().cpu().clone().detach()
            
            total_correct += correct_mask.sum().cpu().item()
            total_num += ims.shape[0]

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if scheduler is not None: scheduler.step()
    
    acc = total_correct / total_num
    loss = total_loss / total_num
    
    train_ret = {}

    train_ret["accuracy"], train_ret["loss"], train_ret["acc_mask"], train_ret["conf_mask"] = acc, loss, mask_after_opt, conf_after_opt 
    
    return train_ret



def train(model, loader, opt, scheduler, loss_fn, EPOCHS, patience = 5, eval_every = False, eval_loader= None):
    stop_train_patience = 0
    mask_list = []
    conf_list = []
    mask_list_tr = []
    conf_list_tr = []
    mask_after_opt_list = [] #this one is used for getting forgetting counts
    conf_after_opt_list = [] 

    for ep in range(EPOCHS):
        if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
            # decay LR on Plateau. Metric based on training loss rather than validation 
            # since we are often learning on random labels in these experiments
            train_ret = single_epoch(model, loader, opt, loss_fn, None)
            scheduler.step(train_ret["loss"]) 
        
        else:
            train_ret = single_epoch(model, loader, opt, loss_fn, scheduler)
            # print("LR", opt.param_groups[0]['lr'])

        acc, loss, mask_after_opt, conf_after_opt = train_ret["accuracy"], train_ret["loss"], train_ret["acc_mask"], train_ret["conf_mask"]
        
        mask_after_opt_list.append(mask_after_opt.unsqueeze(0))
        conf_after_opt_list.append(conf_after_opt.unsqueeze(0))

        if eval_every:
            # eval_ret =  eval(model, eval_loader, eval_mode = True)
            # mask_list.append(eval_ret["acc_mask"].unsqueeze(0))
            # conf_list.append(eval_ret["conf_mask"].unsqueeze(0))

            # print(f'Epoch: {ep+1} | Eval Loader Accuracy: {eval_ret["accuracy"]:.4f}%')

            eval_ret =  eval(model, eval_loader, eval_mode = False)
            mask_list_tr.append(eval_ret["acc_mask"].unsqueeze(0))
            conf_list_tr.append(eval_ret["conf_mask"].unsqueeze(0))

            print(f'Epoch: {ep+1} | Train Mode Eval Loader Accuracy: {eval_ret["accuracy"]:.4f}%')

        if acc == 1.0: stop_train_patience += 1
        
        if stop_train_patience == patience: 
            print(f'Epoch: {ep+1} | Accuracy: {acc * 100:.4f}% | Loss: {loss:.2e}')
            break
        if (ep+1)%5 == 0 or (ep+1)==EPOCHS:
            print(f'Epoch: {ep+1} | Accuracy: {acc * 100:.4f}% | Loss: {loss:.2e}')
        
    
    return_dict = {}
    # return_dict["acc_mask"] = torch.cat(mask_list) if mask_list != [] else None
    # return_dict["conf_mask"] = torch.cat(conf_list) if conf_list != [] else None

    return_dict["acc_mask"] = torch.cat(mask_list_tr) if mask_list_tr != [] else None
    return_dict["conf_mask"] = torch.cat(conf_list_tr) if conf_list_tr != [] else None

    return_dict["acc_mask_after_opt"] = torch.cat(mask_after_opt_list) if mask_after_opt_list != [] else None
    return_dict["conf_mask_after_opt"] = torch.cat(conf_after_opt_list) if conf_after_opt_list != [] else None

    return return_dict

def eval(model, loader, eval_mode = True):
    if eval_mode: model.eval()
    else: model.train()
    mask = torch.zeros(len(loader.dataset))
    conf = torch.zeros(len(loader.dataset))
    with torch.no_grad():
        total_correct, total_num = 0., 0.
        for ims, labs, ids in loader:
            ims, labs = ims.cuda(), labs.cuda()
            with autocast():
                out = model(ims)

                correct_mask = out.argmax(1).eq(labs)
                conf_mask = out[torch.arange(labs.shape[0]),labs]
                mask[ids.squeeze(-1)] = correct_mask.float().cpu()
                conf[ids.squeeze(-1)] = conf_mask.float().cpu()

                total_correct += correct_mask.sum().cpu().item()
                total_num += ims.shape[0]

    ret = {}
    ret["accuracy"] = total_correct / total_num * 100
    ret["acc_mask"] = mask
    ret["conf_mask"] = conf 

    return ret


def get_forgetting_counts(masks):
    num_examples = masks.shape[1]
    num_epochs = masks.shape[0]
    mask2 = torch.ones((num_epochs, num_examples))
    # mask2 represents the accuracy for the same example at the next epoch
    # if mask1 is greater than mask2 then a forgetting event happened
    mask2[:-1] = masks[1:]
    diff_mask = masks - mask2
    diff_mask[diff_mask > 0] = 1
    diff_mask[diff_mask != 1] = 0
    num_forgetting_events = diff_mask.sum(dim = 0)
    return num_forgetting_events

def get_first_epoch_where_we_learn_forever(mask):
    #never forget once you learned
    # Example:
    # >>> a = torch.tensor([0,0,1,1,0,1,1,1,1])
    # >>> z  = torch.flip(a, [0])
    # >>> z
    # tensor([1, 1, 1, 1, 0, 1, 1, 0, 0])
    # >>> z.argmax()
    # tensor(0)
    # >>> z.argmin()
    # tensor(4)

    # What if example is correct from the beginning? 
    # >> Just add an extra row at the top that has all 0s

    # What if we never overfit on that sample even after many epochs?
    # >> Just add an extra row at the bottom that has all 1s
    mask = torch.cat([torch.zeros(1, mask.shape[1]), mask, torch.ones(1, mask.shape[1])])
    z  = torch.flip(mask, [0])
    mins = z.argmin(dim = 0)
    total_epochs = mask.shape[0]

    
    return (total_epochs - mins).float().numpy()

def get_first_epoch_where_we_forget(mask):
    # Example:
    # >>> a = torch.tensor([1,1,0,0,1,1,0,0,0,0])
    # >>> b = torch.tensor([1,1,1,1,1,1])
    # >>> a.argmin()
    # tensor(2)
    # What if example was never learnt? 
    # Just add an extra row at the top that has all 1s

    # What if example is never forgotten? 
    # Just add an extra row at the bottom that has all 0s
    mask = torch.cat([torch.ones(1, mask.shape[1]), mask, torch.zeros(1, mask.shape[1])])
    return mask.argmin(dim = 0).float().numpy()

def get_first_epoch_where_we_forget_forever(mask):
    #this is same as get first epoch where we learn forever, 
    # except that we reverse the masks
    return get_first_epoch_where_we_learn_forever(1 - mask)

