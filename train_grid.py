import os, sys, random

from utils import *
from models import *
from dataloader import *
import params, json







def trainer(args):
    pre_dict, ft_dict = return_loaders(args)

    from torch.optim import SGD
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0)
    model_list = ["resnet-9", "lenet"]
    sched_list = ["triangle", "step"]
    lr_list = [0.1, 0.05, 0.01, 0.005, 0.001]
    preloader = pre_dict["train_loader"]
    ftloader = ft_dict["train_loader"]

    num_repeats = 50
    
    for i in range(num_repeats):
        m_type, s_type, l_type = random.choice(model_list), random.choice(sched_list), random.choice(lr_list)
        print(i, m_type, s_type, l_type)
        model = get_model(m_type, in_channels=1)
        loader = preloader
        optimizer = SGD(model.parameters(), lr=l_type, momentum=0.9, weight_decay=5e-4)
        scheduler, EPOCHS = get_scheduler_epochs(s_type, optimizer, loader)

        masks = train(model, loader, optimizer, scheduler, loss_fn, EPOCHS = EPOCHS,
            eval_every = True, eval_loader = preloader)
        
        torch.save(masks, f"masks/{args.dataset1}/learn_standard_{i}.pt")

        # Stage 2
        loader = ftloader
        optimizer = SGD(model.parameters(), lr=l_type, momentum=0.9, weight_decay=5e-4)
        scheduler, EPOCHS = get_scheduler_epochs(s_type, optimizer, loader)

        masks = train(model, loader, optimizer, scheduler, loss_fn, EPOCHS = EPOCHS,
            eval_every = True, eval_loader = preloader)
        
        torch.save(masks, f"masks/{args.dataset1}/forget_standard_{i}.pt")


if __name__ == "__main__":
    parser = params.parse_args()
    args = parser.parse_args()
    args = params.add_config(args) if args.config_file != None else args
    args = vars(args)
    print(args)
    trainer(args)