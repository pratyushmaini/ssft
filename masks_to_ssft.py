from utils import *
import sys
import glob
from dataloader import *
import ipdb
import params


#this code combines multiple forgetting masks for first and second split, and then provides one tensor of ssft for the entire dataset
def ssft_calculator(args):
    mask_dir = f"masks/{args['dataset1']}/"

    #list all forget files in mask_dir
    mask_files = glob.glob(mask_dir + "standard/*forget*")

    f_e = []
    for file in mask_files:
        mask = torch.load(file)
        mask = mask["acc_mask"]
        forget_epochs = get_first_epoch_where_we_forget_forever(mask)
        f_e.append(torch.from_numpy(forget_epochs).unsqueeze(0))

    f_e = torch.cat(f_e)
    f_e_split_1 = f_e.mean(dim = 0)

    #now do the same for the second split masks
    mask_files = glob.glob(mask_dir + "reverse/*forget*")

    f_e = []

    for file in mask_files:
        mask = torch.load(file)
        mask = mask["acc_mask"]
        forget_epochs = get_first_epoch_where_we_forget_forever(mask)
        f_e.append(torch.from_numpy(forget_epochs).unsqueeze(0))

    f_e = torch.cat(f_e)
    f_e_split_2 = f_e.mean(dim = 0)

    #now combine based on example ids
    num_examples = f_e_split_1.shape[0] + f_e_split_2.shape[0]
    pre_indices, ft_indices =  get_split_ids(num_examples, ratio = 0.5)
    f_e = torch.zeros(num_examples)
    f_e[pre_indices] = f_e_split_1
    f_e[ft_indices] = f_e_split_2
    torch.save(f_e, f"{mask_dir}/ssft_{args['dataset1']}.pt")



if __name__ == "__main__":
    parser = params.parse_args()
    args = parser.parse_args()
    args = params.add_config(args) if args.config_file != None else args
    args = vars(args)
    args["dataset2"] = args["dataset1"]
    args["noise_2"] = args["noise_1"]
    print(args)
    seed_everything(args['seed'])
    ssft_calculator(args)
