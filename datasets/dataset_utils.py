from cgi import print_arguments
from copyreg import pickle
import imp
# from msilib.schema import Error
from multiprocessing.spawn import import_main_path
from pathlib import Path
import numpy as np
import torch
import os

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
import shutil
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision import datasets, transforms
from typing import Callable, Optional, Tuple, Any
from flwr.dataset.utils.common import create_lda_partitions

from . import personalization_cifar as p_cifar
from . import personalization_celeba as p_celeba
from .speech_commands import get_speechcommands_and_partition_it, raw_audio_to_AST_spectrogram, PartitionedSPEECHCOMMANDS
from .leaf.pickle_dataset import PickleDataset

def get_dataloader(
    dataset_name: str, 
    path_to_data: str, 
    anywhere: bool,
    cid: str, 
    is_train: bool, 
    batch_size: int, 
    workers: int, 
    preprocess,
    args
):
    """Generates trainset/valset object and returns appropiate dataloader."""

    collate_fn = None
    if dataset_name == "speechcommands":
        # ! not we might want to do a custom weighed sampler to deal with sever class imbalance in SC
        partition = "training"
        path_to_client_data = Path(path_to_data)/str(cid)
        dataset = PartitionedSPEECHCOMMANDS(path_to_client_data, subset=partition, transforms=raw_audio_to_AST_spectrogram(), classes=args.num_classes, wav2fbank=True)
        collate_fn = dataset._collate_fn
    elif dataset_name == 'femnist':
        partition = "train" if is_train else "test"
        pdataset = PickleDataset(dataset_name=dataset_name, pickle_root=path_to_data)
        raw_data = pdataset.get_dataset_pickle(dataset_type=partition, client_id=cid)
        dataset = TorchVision_FL(dataset_name=dataset_name, data=raw_data, transform=preprocess)
    else:
        # dataset = load_dataset(dataset_name, Path(path_to_data), cid, partition, preprocess)
        partition = "train" if is_train else "val"
        path_to_data = Path(path_to_data) / cid / (partition + ".pt")
        dataset = TorchVision_FL(dataset_name=dataset_name, path_to_data=path_to_data, transform=preprocess)
    
    # we use as number of workers all the cpu cores assigned to this actor
    kwargs = {"num_workers": workers, "pin_memory": True, "drop_last": False}
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, **kwargs)


def get_random_id_splits(total: int, val_ratio: float, shuffle: bool = True):
    """splits a list of length `total` into two following a
    (1-val_ratio):val_ratio partitioning.

    By default the indices are shuffled before creating the split and
    returning.
    """

    if isinstance(total, int):
        indices = list(range(total))
    else:
        indices = total

    split = int(np.floor(val_ratio * len(indices)))
    # print(f"Users left out for validation (ratio={val_ratio}) = {split} ")
    if shuffle:
        np.random.shuffle(indices)
    return indices[split:], indices[:split]

def do_fl_partitioning(
    dataset_name, 
    path_to_dataset, 
    dataset_info, 
    pool_size, 
    alpha,
    anywhere, 
    num_classes, 
    val_ratio=0.0):

    partitions, _ = create_lda_partitions(
        dataset_info, num_partitions=pool_size, concentration=alpha, accept_imbalanced=True
    )

    # Show label distribution for first partition (purely informative)
    partition_zero = partitions[0][1]
    hist, _ = np.histogram(partition_zero, bins=list(range(num_classes + 1)))
    print(
        f"Class histogram for 0-th partition (alpha={alpha}, {num_classes} classes): \n {hist}"
    )

    # now save partitioned dataset to disk
    # first delete dir containing splits (if exists), then create it
    splits_dir = Path(path_to_dataset) / f"federated_{alpha}"
    if splits_dir.exists() and os.listdir(splits_dir):
        print('Dataset has been generated.')
        return splits_dir
        # shutil.rmtree(splits_dir)
        
    Path.mkdir(splits_dir, parents=True)

    print('LDA partition with alpha{}:'.format(alpha))
    for p in tqdm(range(pool_size)):

        labels = partitions[p][1]
        imgs = partitions[p][0]
        
        if anywhere:
            try:
                if 'cifar' in dataset_name:
                    imgs = p_cifar.personalize_cifar(imgs)
                elif dataset_name == 'celeba':
                    p_celeba.personalize_celeba(path=splits_dir.parent, alpha=alpha, data=imgs)
                else:
                    raise NotImplementedError
            except:
                shutil.rmtree(splits_dir)
                raise ValueError('occurr error when generating anywhere data.')
            
        # create dir
        Path.mkdir(splits_dir / str(p))

        if val_ratio > 0.0:
            # split data according to val_ratio
            train_idx, val_idx = get_random_id_splits(len(labels), val_ratio)
            val_imgs = imgs[val_idx]
            val_labels = labels[val_idx]

            with open(splits_dir / str(p) / "val.pt", "wb") as f:
                torch.save([val_imgs, val_labels], f)

            # remaining images for training
            imgs = imgs[train_idx]
            labels = labels[train_idx]

        with open(splits_dir / str(p) / f"train.pt", "wb") as f:
            torch.save([imgs, labels], f)

    return splits_dir

class TorchVision_FL(VisionDataset):
    """This is just a trimmed down version of torchvision.datasets.MNIST.

    Use this class by either passing a path to a torch file (.pt)
    containing (data, targets) or pass the data, targets directly
    instead.
    """

    def __init__(
        self,
        dataset_name,
        path_to_data=None,
        data=None,
        anywhere=False,
        transform: Optional[Callable] = None,
    ) -> None:
        self.path = path_to_data.parent if path_to_data else None
        super(TorchVision_FL, self).__init__(self.path, transform=transform)
        self.anywhere = anywhere
        self.transform = transform
        self.dataset_name = dataset_name

        if 'cifar' in dataset_name:
            # load data and targets (path_to_data points to an specific .pt file)
            self.data, self.targets = torch.load(path_to_data)
        elif dataset_name == 'femnist':
            self.data, self.targets = data['data'], data['label']
        elif dataset_name == 'celeba':
            self.filename, self.targets = torch.load(path_to_data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if 'cifar' in self.dataset_name:
            img, target = self.data[index], int(self.targets[index])
        elif self.dataset_name == 'femnist':
            img, target = self.data[index].reshape(28, 28, -1), int(self.targets[index])
            img = img.expand(-1, -1, 3) # input channel is fixed as 3
        elif self.dataset_name == 'celeba':
            if self.anywhere:
                path = os.path.join(self.path.parent, 'transformed_data', self.filename[index])
            else:
                path = os.path.join(self.path.parents[1], 'celeba', 'img_align_celeba', self.filename[index])
            img = Image.open(path).convert('RGB')
            target = int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if not isinstance(img, Image.Image):  # if not PIL image
            if not isinstance(img, np.ndarray):  # if torch tensor
                img = img.numpy()
                
                if self.dataset_name == 'femnist':
                    img = (img * 255).astype(np.uint8)

            img = Image.fromarray(img)
            
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self) -> int:
        if self.dataset_name == 'cifar100' or self.dataset_name == 'femnist':
            return len(self.data)
        elif self.dataset_name == 'celeba':
            return len(self.filename)
        else:
            raise NotImplementedError


def get_transform(args):

    if args.dataset == 'speechcommands':
        transform_train = transform_test = raw_audio_to_AST_spectrogram()
        
    elif args.dataset == 'femnist':
        mean = [0.1307, 0.1307, 0.1307]
        std = [0.3081, 0.3081, 0.3081]

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop((args.input_size, args.input_size), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        mean = [x/255.0 for x in [129.37731888,  124.10583864, 112.47758569]]
        std = [x/255.0 for x in [68.20947949,  65.43124043,  70.45866994]]

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop((args.input_size, args.input_size), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    return transform_train, transform_test


def getCIFAR10(path_to_data="./data", preprocess_train=None, preprocess_test=None):
    """Downloads CIFAR10 dataset and generates a unified training set (it will
    be partitioned later using the LDA partitioning mechanism."""

    # download dataset and load train set
    train_set = datasets.CIFAR10(root=path_to_data,
                                 train=True,
                                 download=True,
                                 transform=preprocess_train
                                 )

    train_info = [train_set.data, np.array(train_set.targets)]

    test_set = datasets.CIFAR10(root=path_to_data, 
                                train=False,
                                transform=preprocess_test
                                )

    # returns path where training data is and testset
    return train_info, test_set


def getCIFAR100(path_to_data="./data", preprocess_train=None, preprocess_test=None):
    """Downloads CIFAR100 dataset and generates a unified training set (it will
    be partitioned later using the LDA partitioning mechanism."""

    # download dataset and load train set
    train_set = datasets.CIFAR100(root=path_to_data,
                                  train=True,
                                  download=True,
                                  transform=preprocess_train
                                  )

    train_info = [train_set.data, np.array(train_set.targets)]

    test_set = datasets.CIFAR100(
        root=path_to_data,
        train=False,
        transform=preprocess_test
    )

    # returns path where training data is and testset
    return train_info, test_set

def getCelebA(path_to_data='./data', preprocess_train=None, preprocess_test=None):
    label_transform = lambda x : x[31] # labels of smile
    train_set = datasets.CelebA(root=path_to_data,
                                split='train',
                                target_type='attr',
                                transform=preprocess_train,
                                target_transform=label_transform
                                )
    
    train_info = [np.array(train_set.filename), np.array(train_set.attr[:, 31])] # smiling classification
    
    test_set = datasets.CelebA(root=path_to_data,
                               split='test',
                               target_type='attr',
                               transform=preprocess_test,
                               target_transform=label_transform
                               )
    return train_info, test_set

def getDataset(args):

    data_path = args.data_path
    # We can overwrite the default one, e.g. for ViT models it will be different from default.
    _, transform_test = get_transform(args)

    fed_dir = None
    training_data = None
    if args.dataset == 'cifar10':
        training_data, test_set = getCIFAR10(data_path, preprocess_test=transform_test)
    elif args.dataset == 'cifar100':
        training_data, test_set = getCIFAR100(data_path, preprocess_test=transform_test)
    elif args.dataset == 'celeba':
        training_data, test_set = getCelebA(data_path, preprocess_test=transform_test)
    elif args.dataset == 'femnist':
        fed_dir = data_path
        pdataset = PickleDataset(dataset_name=args.dataset, pickle_root=data_path)
        raw_data = pdataset.get_dataset_pickle(dataset_type='test')
        test_set = TorchVision_FL(dataset_name=args.dataset, data=raw_data, transform=transform_test)
    elif args.dataset == 'speechcommands':
        # download and parition (will do nothing if partitions are found)
        # this will generate explicit data paritions inside <data_path>/speechcomands/federated/<here individual directories for each client>
        fed_dir = get_speechcommands_and_partition_it(data_path, version=2)

        # as validation set we'll fuse the data from validation clients into a single big set
        subset = "validation" # use "testing" for test set
        path_to_global_split = Path(fed_dir).parent
        test_set = PartitionedSPEECHCOMMANDS(path_to_global_split, subset=subset, transforms=raw_audio_to_AST_spectrogram(), classes=args.num_classes, wav2fbank=True)
    else:
        raise ValueError('This dataset is not included!')

    return training_data, test_set, fed_dir
