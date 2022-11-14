


from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils import test, train
from datasets.speech_commands import PartitionedSPEECHCOMMANDS, raw_audio_to_AST_spectrogram


def evaluate(val_dataset, model, args):
    loader = DataLoader(val_dataset, batch_size=20, shuffle=False,
                        num_workers=4, collate_fn=val_dataset._collate_fn, pin_memory=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    acc, loss = test(model, loader, device, args)

    return acc

def fine_tune(train_dataset, model, client_layer, lr, args):
    loader = DataLoader(train_dataset, batch_size=20, num_workers=4, collate_fn=train_dataset._collate_fn, pin_memory=True, drop_last=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    train(model, client_layer, loader, lr, args.local_epoch, 1, device, args)

def load_pretrained_model(model, args):
    print(f"Loading checkpoint from: {args.personal_model}")
    model_saved_data = torch.load(args.personal_model)
    model.load_state_dict(model_saved_data['model_state_dict'])

def eval_pretrained_on_all_clients(fed_dir:Path, model, args, num_clients:int=256):

    total_clients = 256
    with open(
            f'acc_{args.mode}_FreezeBase_{args.freeze_base}_FreezeMode_{args.personal_freeze_mode}_OnlyToken_{args.personal_only_token}_PersonalFinetuning_{args.personal_fine_tune}_BkgV_{args.bkg_volume}_BkgF_{args.bkg_frequency}.txt',
            'w') as f:
        for cid in range(total_clients):

            # this defines up to which layer we can run the model
            client_layer = cid%12 # fix exit layer for each client (similarly as it's done in `main.py` before creating the Clients)
            # print(f"Client {cid} trains up to layer {client_layer}")

            data_path = fed_dir / str(cid)

            # ! pass args.num_classes to classes input argument (but do this once new model is trained)
            dataset = PartitionedSPEECHCOMMANDS(data_path, "training", transforms=raw_audio_to_AST_spectrogram(),
                                                classes=12, wav2fbank=True, bkg_volume=args.bkg_volume,
                                                bkg_frequency=args.bkg_frequency)
            _, hist = dataset.get_balanced_sampler()

            # discard those clients that have very few data points
            if sum(hist) > 10:
                # partition
                train_dataset, val_dataset = dataset.split(ratio=0.7)  # < ------------------------ repalce this with whatever you want
                assert len(train_dataset) + len(val_dataset) == sum(hist)
                # print some statistics
                # print(f"cid {cid} --> total datapoints: {sum(hist)} (train: {len(train_dataset)} / eval: {len(val_dataset)}) --> Histogram of labels: {hist}")
                print(f'cid {cid}')

                # fine-tuning
                if args.personal_fine_tune:
                    acc_list = []
                    acc_mean_list = []
                    lr_list = eval(args.lr_list)
                    for lr in lr_list:
                        # load pre-trained model
                        load_pretrained_model(model, args)

                        # freeze entire accumulator (either linear/mlp/accumulator mode works )
                        if args.personal_freeze_mode:
                            for param in model.accumulator.parameters():
                                param.requires_grad = False

                        # freeze all layers except token
                        if args.personal_only_token:
                            for param in model.parameters():
                                param.requires_grad = False

                            model.accumulator.cls_token.requires_grad = True

                        fine_tune(train_dataset, model, client_layer, lr, args)

                        # do eval again after fine-tuning
                        acc = evaluate(val_dataset, model, args)
                        acc_list.append(acc)
                        acc_mean = sum(acc) / len(acc)
                        acc_mean_list.append(acc_mean)

                    acc = acc_list[acc_mean_list.index(max(acc_mean_list))]

                else:
                    # just do eval
                    acc = evaluate(val_dataset, model, args)

                # find layer index with max acc.
                indices = [i for i, x in enumerate(acc) if x == max(acc)]
                f.write(f'{cid}\t{list(acc)}\t{indices}\n')


