from utils import *
from models import *
from dataloader import *
import params



def trainer(args):
    pre_dict, ft_dict = return_loaders(args)
    in_channels = 1 if args["dataset1"] in ["mnist","fashionmnist","emnist"] else 3

    from torch.optim import SGD
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0)

    preloader = pre_dict["train_loader"]
    ftloader = ft_dict["train_loader"]

    if args["reverse_splits"]:
        #This is for the case where we want to compute metrics on the second split (and then combine for both the splits)
        preloader, ftloader = ftloader, preloader

    m_type, s_type, l_type = args['model_type'], args['sched'], args['lr1']
    root = f"masks/{args['dataset1']}/{args['name']}"
    os.makedirs(root, exist_ok=True)
    
    torch.manual_seed(args['model_seed'])
    model = get_model(m_type, in_channels=in_channels)
    
    loader = preloader
    optimizer = SGD(model.parameters(), lr=l_type, momentum=0.9, weight_decay=5e-4)
    scheduler, EPOCHS = get_scheduler_epochs(s_type, optimizer, loader)

    masks = train(model, loader, optimizer, scheduler, loss_fn, EPOCHS = EPOCHS,
        eval_every = True, eval_loader = preloader)
    
    torch.save(masks, f"{root}/learn_{m_type}_{l_type}_{s_type}_{args['model_seed']}.pt")

    # Stage 2
    loader = ftloader
    optimizer = SGD(model.parameters(), lr=l_type, momentum=0.9, weight_decay=5e-4)
    scheduler, EPOCHS = get_scheduler_epochs(s_type, optimizer, loader)

    masks = train(model, loader, optimizer, scheduler, loss_fn, EPOCHS = EPOCHS,
        eval_every = True, eval_loader = preloader)
    
    torch.save(masks, f"{root}/forget_{m_type}_{l_type}_{s_type}_{args['model_seed']}.pt")


if __name__ == "__main__":
    parser = params.parse_args()
    args = parser.parse_args()
    args = params.add_config(args) if args.config_file != None else args
    args = vars(args)
    args["dataset2"] = args["dataset1"]
    args["noise_2"] = args["noise_1"]
    print(args)
    seed_everything(args['seed'])
    trainer(args)