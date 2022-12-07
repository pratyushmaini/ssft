from utils import *
from models import *
from dataloader import *
import params

def trainer(args):
    pre_dict, ft_dict = return_loaders(args)

    from torch.optim import SGD
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0)
    preloader = pre_dict["train_loader"]
    ftloader = ft_dict["train_loader"]


    m_type, s_type, l_type = args['model_type'], args['sched'], args['lr1']
    model = get_model(m_type, in_channels=1)
    loader = preloader
    optimizer = SGD(model.parameters(), lr=l_type, momentum=0.9, weight_decay=5e-4)
    scheduler, EPOCHS = get_scheduler_epochs(s_type, optimizer, loader)

    masks = train(model, loader, optimizer, scheduler, loss_fn, EPOCHS = EPOCHS,
        eval_every = True, eval_loader = preloader)
    
    torch.save(masks, f"masks/{args['dataset1']}/learn_standard_{args['model_id']}.pt")

    # Stage 2
    loader = ftloader
    optimizer = SGD(model.parameters(), lr=l_type, momentum=0.9, weight_decay=5e-4)
    scheduler, EPOCHS = get_scheduler_epochs(s_type, optimizer, loader)

    masks = train(model, loader, optimizer, scheduler, loss_fn, EPOCHS = EPOCHS,
        eval_every = True, eval_loader = preloader)
    
    torch.save(masks, f"masks/{args['dataset1']}/forget_standard_{args['model_id']}.pt")


if __name__ == "__main__":
    parser = params.parse_args()
    args = parser.parse_args()
    args = params.add_config(args) if args.config_file != None else args
    args = vars(args)
    args["dataset2"] = args["dataset1"]
    args["noise_2"] = args["noise_1"]
    print(args)
    trainer(args)