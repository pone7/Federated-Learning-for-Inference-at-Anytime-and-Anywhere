import argparse
from ast import arg
from pathlib import Path
from datetime import datetime
import flwr as fl
from flwr.common.typing import Scalar
import ray
import json
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple
from datasets.dataset_utils import getDataset, do_fl_partitioning, get_dataloader, get_transform
from utils import train, test, cosine_decay_with_warmup, construct_output_dir
from typing import Dict, List, Tuple, Optional
from strategy import opt_strategy
from models import get_model, get_accumulator

import copy
from thop import clever_format

USE_FEDBN=False
ROUND=0
MEAN_ACC=0

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments

    # general
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--print_interval", type=int, default=10)
    parser.add_argument('--output_folder', type=str, required=False)
    parser.add_argument("--dataset", type=str, default='cifar100')
    parser.add_argument("--data_path", type=str, default='/usr/data/cifar100')
    parser.add_argument("--no_aug", type=bool, default=True, help='timm transformation args')
    parser.add_argument("--auto_output_dir", action='store_true', help='will ignore `--output_folder` and instead define it at runtime based on the passed input arguments')

    # model
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument("--base_model", type=str, default='deit_base_patch16')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--input_size", type=int, default=224, help='timm transformation args')
    parser.add_argument('--num_classes', default=100, type=int)
    parser.add_argument('--in_chans', default=3, type=int, help="input channels to model (3 for images, 1 for speechcommands)")
    parser.add_argument('--freeze_base', action='store_true')

    # optimier
    parser.add_argument('--clip_grad', type=float, required=False, default=0.0)
    parser.add_argument('--lr', type=float, required=False, default=0.001)
    parser.add_argument('--warmup_lr', type=float, required=False, default=1e-5)
    parser.add_argument('--warmup_epochs', type=float, required=False, default=0)
    parser.add_argument('--lr_min', type=float, required=False, default=1e-5)
    parser.add_argument('--momentum', type=float, required=False, default=0.9)
    parser.add_argument('--wd', type=float, required=False, default=0.0)
    parser.add_argument('--optim', type=str, required=False, default='SGD')

    # FL
    parser.add_argument("--user_num", type=int, default=100)
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--local_epoch", type=int, default=1)
    parser.add_argument("--local_bs", type=int, default=20)
    parser.add_argument("--test_bs", type=int, default=512)
    parser.add_argument("--fit_frac", type=float, default=0.1, help='sample fraction of users each round.')
    parser.add_argument("--alpha", type=float, default=1000.0, help='LDA data partition param determining the non IID-ness')
    parser.add_argument("--strategy", type=str, default='FedAvg')
    parser.add_argument('--anywhere', action='store_true')
    parser.add_argument('--anytime', action='store_true') 
    parser.add_argument('--multi_tier', action='store_true') 
    parser.add_argument('--personalization', type=str, default=None) 

    # Accumulator
    parser.add_argument('--feature_size', default=12, type=int, help='num of cls tokens from pretrained model')
    parser.add_argument('--recurrent_steps', default=1, type=int, help='num of steps')
    parser.add_argument('--heads', default=12, type=int)
    parser.add_argument('--mlp_dim', default=1536, type=int)
    parser.add_argument('--dim', default=384, type=int)
    parser.add_argument('--depth', default=1, type=int, help='num of layers of accumulator')
    parser.add_argument('--adpffn', action='store_true')
    parser.add_argument('--replace', action='store_true')
    
    parser.add_argument('--fine_tune_mlp', action='store_true', help='replace accumulator with layer-wise linear head.')
    parser.add_argument('--mode', choices=['linear', 'mlp', 'accumulator', 'finetune'],default='accumulator')

    # flower simulation 
    parser.add_argument("--num_client_cpus", type=int, default=1, help='CPU workers per client.')
    parser.add_argument("--num_gpus", type=float, default=1.0, help='proportion of usage per GPU for each client.')
    parser.add_argument("--device", type=str, default='cuda:0')

    # personalisation
    parser.add_argument("--personal_commands", action='store_true', help='Loads a pre-trained model and does personalisation on speechcommands (no FL)')
    parser.add_argument("--personal_model", type=str, help='path to a trained model resides (secenario 1, 2 or 3) -- ensure the relevant arguments (e.g. model related) remain the same')
    parser.add_argument("--bkg_volume", type=float, default=0.1, help='background noise volume.')
    parser.add_argument("--bkg_frequency", type=float, default=0.8, help='background noise frequency.')
    parser.add_argument("--lr_list", type=str, default="[0.1]", help='learning rate searching for fine-tuning.')
    parser.add_argument("--personal_fine_tune", action='store_true', help='fine-tune for personalisation.')
    parser.add_argument("--personal_freeze_mode", action='store_true', help='only fine-tune the adaptor.')
    parser.add_argument("--personal_only_token", action='store_true', help='only fine-tune the token.')
    args = parser.parse_args()
    return args

# parse input arguments
args = args_parser()

# Flower client that will be spawned by Ray
# Adapted from Pytorch quickstart example
class Client(fl.client.NumPyClient):
    def __init__(self, cid: str, lid: str, model: nn.Module, fed_dir_data: str, args):
        self.cid = cid
        self.lid = lid
        self.fed_dir = Path(fed_dir_data)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

        # determine device
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # instantiate model
        self.model = copy.deepcopy(model)
        self.preprocess_train, self.preprocess_test = get_transform(args)
        self.args = args

    def get_parameters(self):
        if USE_FEDBN:
            # Return model parameters as a list of NumPy ndarrays, excluding parameters of BN layers when using FedBN
            return [
                val.cpu().numpy()
                for name, val in self.model.state_dict().items()
                if "bn" not in name
            ]
        else:
            # Return model parameters as a list of NumPy ndarrays
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    # def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
    def get_properties(self, ins):
        return self.properties

    def set_parameters(self, parameters):
        if USE_FEDBN:
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        trainloader = get_dataloader(
            self.args.dataset,
            self.fed_dir,
            self.args.anywhere,
            self.cid,
            is_train=True,
            batch_size=int(config["batch_size"]),
            workers=num_workers,
            preprocess=self.preprocess_train,
            args=self.args
        )
        # send model to device
        self.model.to(self.device)
        
        lr = cosine_decay_with_warmup(ROUND, args.lr, args.num_rounds, hold_base_rate_steps=1e-5)
        
        # train
        train(self.model, self.lid, trainloader, lr, epochs=int(config["epochs"]), round=ROUND, device=self.device, args=self.args)

        # return local model and statistics
        return self.get_parameters(), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):

        # print(f"evaluate() on client cid={self.cid}")
        self.set_parameters(parameters)

        # load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        valloader = get_dataloader(
            self.args.dataset,
            self.fed_dir, 
            self.args.anywhere,
            self.cid, 
            is_train=False, 
            batch_size=args.test_bs, 
            workers=num_workers,
            preprocess=self.preprocess_test,
            args=self.args
        )
        
        # send model to device
        self.model.to(self.device)

        # evaluate
        accuracy, loss = test(self.model, valloader, device=self.device, args=args)

        print(accuracy[self.lid], loss[self.lid])
        
        with open(f"{self.args.output_folder}/results_local.txt", mode="a") as eval_file:
            eval_file.write('lid {}: accuracy {}, loss {}'.format(self.lid, accuracy[self.lid], loss[self.lid]))
            eval_file.write("\n")

        # return statistics
        return float(loss[self.lid]), len(valloader.dataset), {"accuracy": float(accuracy[self.lid])}


def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "epochs": str(args.local_epoch),  # number of local epochs
        "batch_size": str(args.local_bs),
    }
    return config


def get_parameters(net) -> List[np.ndarray]:
    # net.train()
    if USE_FEDBN:
        # Return model parameters as a list of NumPy ndarrays, excluding parameters of BN layers when using FedBN
        return [
            val.cpu().numpy()
            for name, val in net.state_dict().items()
            if "bn" not in name
        ]
    else:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    # net.train()
    if USE_FEDBN:
        keys = [k for k in net.state_dict().keys() if "bn" not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=False)
    else:
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)


def get_eval_fn(
    testset, model, args, tb_writter,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        global ROUND
        global MEAN_ACC

        if ROUND % 10 == 0:
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")

            set_parameters(model, weights)
            model.to(device)

            collate = testset._collate_fn if args.dataset=="speechcommands" else None
            testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_bs, num_workers=4, collate_fn=collate)
            acc, loss = test(model, testloader, device, args)

            lr = cosine_decay_with_warmup(ROUND, args.lr, args.num_rounds, args.warmup_lr, args.warmup_epochs, args.lr_min)
            with open(f"{args.output_folder}/results_global.txt", mode="a") as eval_file:
                eval_file.write('Round{}: \n lr: {:4f}, \n acc: {}, mean: {:3f}, \n loss: {}'.format(ROUND, lr, acc, acc.mean(), loss))
                eval_file.write("\n")

            # record to tensorboard (average acc, loss and per-exit acc)
            tb_writter.add_scalar(tag='global/avg_accuracy', scalar_value=acc.mean(), global_step=ROUND)
            tb_writter.add_scalar(tag='global/avg_loss', scalar_value=loss.mean(), global_step=ROUND)
            for i, (exit_acc, exit_loss) in enumerate(zip(acc,loss)):
                tb_writter.add_scalar(tag=f'global_per_layer/accuracy_{i}', scalar_value=exit_acc, global_step=ROUND)
                tb_writter.add_scalar(tag=f'global_per_layer/loss_{i}', scalar_value=exit_loss, global_step=ROUND)

            # model saving
            if acc.mean() > MEAN_ACC:
                MEAN_ACC = acc.mean()
                torch.save({
                            'round': ROUND,
                            'model_state_dict': model.state_dict(),
                            'test_acc': acc,
                            }, args.output_folder + '/model.pt')
            ROUND=ROUND+1
            acc = acc.tolist()

        else:
            loss = [-1]
            acc = [-1]
            ROUND=ROUND+1

        # return statistics 
        return loss[-1], {"avg_acc": acc[-1], 'layer_acc': acc}
    return evaluate


# Start Ray simulation (a _default server_ will be created)
# This example does:
# 1. Downloads Dataset
# 2. Partitions the dataset into N splits, where N is the total number of
#    clients. We refere to this as `pool_size`. The partition can be IID or non-IID
# 3. Starts a Ray-based simulation where a % of clients are sample each round.
# 4. After the M rounds end, the global model is evaluated on the entire testset.
#    Also, the global model is evaluated on the valset partition residing in each
#    client. This is useful to get a sense on how well the global model can generalise
#    to each client's data.
if __name__ == "__main__":
    
    pool_size = args.user_num  # number of dataset partions (= number of total clients)
    client_resources = {
        "num_cpus": args.num_client_cpus,
        "num_gpus": args.num_gpus
    }  # each client will get allocated 1 CPU

    # create the folder for results
    if args.auto_output_dir:
        output_dir = construct_output_dir(args)
    else:
        output_dir = Path(args.output_folder)/str(datetime.now().strftime('%b%d_%H_%M_%S'))
    output_dir.mkdir(parents=True)
    args.output_folder = str(output_dir)
    print(f"Output dir: {args.output_folder}")
    tb_writer = SummaryWriter(output_dir/"tensorboard")
    
    with open(f"{args.output_folder}/args.txt", mode="w") as f:
        json.dump(args.__dict__, f, indent=2)

    # download dataset
    train_info, testset, fed_dir = getDataset(args=args)

    # partition dataset (use a large `alpha` to make it IID;
    # a small value (e.g. 1) will make it non-IID)
    # This will create a new directory called "federated": in the directory where
    # CIFAR-10 lives. Inside it, there will be N=pool_size sub-directories each with
    # its own train/set split.
    if fed_dir is None:
        fed_dir = do_fl_partitioning(
            args.dataset, args.data_path, train_info, pool_size=pool_size,
            alpha=args.alpha, anywhere=args.anywhere, num_classes=args.num_classes, val_ratio=0.2
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    if args.mode == 'linear':
        accumulator = nn.ModuleList([nn.Linear(args.dim, args.num_classes) for _ in range(12)])
    elif args.mode == 'mlp':
        accumulator = nn.ModuleList(
            [nn.Sequential(
             nn.LayerNorm(args.dim),
             nn.Linear(args.dim, args.mlp_dim),
             nn.GELU(),
             nn.Dropout(),
             nn.Linear(args.mlp_dim, args.num_classes)) 
        for _ in range(12)])
    elif args.mode == 'accumulator':
        accumulator = get_accumulator(args)
    else:
        accumulator = None
    
    model = get_model(args, accumulator)
    
    if args.freeze_base:
        for n, p in model.named_parameters():
            if not (('accumulator' in n) or ('adaptmlp' in n)):
                p.requires_grad = False
            
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = clever_format([params], "%.3f")
    print(f'trainable params: {params}')
    
    if args.personal_commands: # enable speechcommands personalisation
        from datasets.speech_commands import get_speechcommands_and_partition_it
        from personal_commands import eval_pretrained_on_all_clients
        version = 2
        # this will generate a directory nameed `federated_val` inside the speechcommands directory
        # containing 256 sub-directories, one for each client.
        fed_dir = get_speechcommands_and_partition_it(args.data_path, version=version, split_federated="val")
        print(f"{fed_dir = }")

        # do personalisation
        eval_pretrained_on_all_clients(fed_dir, model, args)
    
    # configure the strategy
    if args.personalization:
        strategy = opt_strategy(args)(
            fraction_fit=args.fit_frac,
            fraction_eval=args.fit_frac, # local evaluation
            min_fit_clients=int(pool_size * args.fit_frac),
            min_available_clients=pool_size,  # All clients should be available
            on_fit_config_fn=fit_config,
            initial_parameters=fl.common.weights_to_parameters(get_parameters(model)),
        )
    else:
        strategy = opt_strategy(args)(
                fraction_fit=args.fit_frac,
                min_fit_clients=int(pool_size * args.fit_frac),
                min_available_clients=pool_size,  # All clients should be available
                on_fit_config_fn=fit_config,
                initial_parameters=fl.common.weights_to_parameters(get_parameters(model)),
                eval_fn=get_eval_fn(testset, model, args, tb_writer),  # centralised testset evaluation of global model
            )
        
        client_layer = {str(i):i%12 for i in range(args.user_num)} # fix exit layer for each client
        
        def client_fn(cid: str):
            lid = client_layer[cid]
            # create a single client instance
            return Client(cid, lid, model, fed_dir, args)

        # (optional) specify ray config
        ray_config = {"include_dashboard": False}

    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        num_rounds=args.num_rounds,
        strategy=strategy,
        ray_init_args=ray_config,
    )
