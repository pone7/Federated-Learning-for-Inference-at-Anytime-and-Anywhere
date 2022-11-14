import imp
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from timm.utils import accuracy


flwr_log = logging.getLogger("flower")

# borrowed from Pytorch quickstart example
def train(net, lid, trainloader, lr, epochs, round, device, args):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    opt = getattr(optim, args.optim)
    optimizer = opt(
         [p for p in net.parameters() if p.requires_grad],
         lr=lr, momentum=args.momentum,
         weight_decay=args.wd
          )

    net.train()
    for _ in range(epochs):
        for iter, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            
            if not args.anytime:
                idx_sample = net.depth - 1

                if args.multi_tier:
                    idx_sample = lid  # fixed exit layer of clients

            if args.dataset == "speechcommands":
                #! this is a temporary work around. The pretrained ViT expects RGB input
                #TODO: here we expand the 2d 1-channel spectrogram. Do this in transforms? or have single 1-ch conv to do the adaptation?
                images = images.expand(-1, 3, images.shape[2], images.shape[3])
                
            outputs = net(images, exit_layer=idx_sample)

            loss = criterion(outputs, labels)
            loss.backward()

            # clip the grad norm to smooth the training
            if args.clip_grad != 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad)
            optimizer.step()

            if iter % args.print_interval == 0:
                print('iter:', iter, 'loss:', loss.item())


# borrowed from Pytorch quickstart example
@torch.no_grad()
def test(net, testloader, device: str, args):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    acc, loss = [], []
    for i, data in enumerate(testloader):
        images, labels = data[0].to(device), data[1].to(device)
        if args.dataset == "speechcommands":
            #! this is a temporary work around. The pretrained ViT expects RGB input
            #TODO: here we expand the 2d 1-channel spectrogram. Do this in transforms? or have single 1-ch conv to do the adaptation?
            images = images.expand(-1, 3, images.shape[2], images.shape[3])

        batch_acc = []
        batch_loss = []
        
        if args.anytime or args.multi_tier:
            for l in range(net.depth):
                outputs = net(images, exit_layer=l)

                batch_loss.append(criterion(outputs, labels).item())
                batch_acc.append(accuracy(outputs, labels)[0].item())
        else:
            outputs = net(images, exit_layer=net.depth-1)

            batch_loss.append(criterion(outputs, labels).item())
            batch_acc.append(accuracy(outputs, labels)[0].item())

        acc.append(batch_acc)
        loss.append(batch_loss)

    acc = np.array(acc).mean(axis=0)
    loss = np.array(loss).mean(axis=0)

    print('test memo used: ', torch.cuda.memory_allocated())
    return np.round(acc, 4), np.round(loss, 4)

def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.
    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """
 
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return float(np.where(global_step > total_steps, 0.0, learning_rate))

'''-------------------------flwr baselines--------------------------'''

from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.random import BitGenerator, Generator, SeedSequence

XY = Tuple[np.ndarray, np.ndarray]
XYList = List[XY]
PartitionedDataset = Tuple[XYList, XYList]

np.random.seed(2020)


def float_to_int(i: float) -> int:
    """Return float as int but raise if decimal is dropped."""
    if not i.is_integer():
        raise Exception("Cast would drop decimals")

    return int(i)


def sort_by_label(x: np.ndarray, y: np.ndarray) -> XY:
    """Sort by label.
    Assuming two labels and four examples the resulting label order
    would be 1,1,2,2
    """
    idx = np.argsort(y, axis=0).reshape((y.shape[0]))
    return (x[idx], y[idx])


def sort_by_label_repeating(x: np.ndarray, y: np.ndarray) -> XY:
    """Sort by label in repeating groups. Assuming two labels and four examples
    the resulting label order would be 1,2,1,2.
    Create sorting index which is applied to by label sorted x, y
    .. code-block:: python
        # given:
        y = [
            0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9
        ]
        # use:
        idx = [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19
        ]
        # so that y[idx] becomes:
        y = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        ]
    """
    x, y = sort_by_label(x, y)

    num_example = x.shape[0]
    num_class = np.unique(y).shape[0]
    idx = (
        np.array(range(num_example), np.int64)
        .reshape((num_class, num_example // num_class))
        .transpose()
        .reshape(num_example)
    )

    return (x[idx], y[idx])


def split_at_fraction(x: np.ndarray, y: np.ndarray, fraction: float) -> Tuple[XY, XY]:
    """Split x, y at a certain fraction."""
    splitting_index = float_to_int(x.shape[0] * fraction)
    # Take everything BEFORE splitting_index
    x_0, y_0 = x[:splitting_index], y[:splitting_index]
    # Take everything AFTER splitting_index
    x_1, y_1 = x[splitting_index:], y[splitting_index:]
    return (x_0, y_0), (x_1, y_1)


def shuffle(x: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle x and y."""
    idx = np.random.permutation(len(x))
    return x[idx], y[idx]


def partition(x: np.ndarray, y: np.ndarray, num_partitions: int) -> List[XY]:
    """Return x, y as list of partitions."""
    return list(zip(np.split(x, num_partitions), np.split(y, num_partitions)))


def combine_partitions(xy_list_0: XYList, xy_list_1: XYList) -> XYList:
    """Combine two lists of ndarray Tuples into one list."""
    return [
        (np.concatenate([x_0, x_1], axis=0), np.concatenate([y_0, y_1], axis=0))
        for (x_0, y_0), (x_1, y_1) in zip(xy_list_0, xy_list_1)
    ]


def shift(x: np.ndarray, y: np.ndarray) -> XY:
    """Shift x_1, y_1 so that the first half contains only labels 0 to 4 and
    the second half 5 to 9."""
    x, y = sort_by_label(x, y)

    (x_0, y_0), (x_1, y_1) = split_at_fraction(x, y, fraction=0.5)
    (x_0, y_0), (x_1, y_1) = shuffle(x_0, y_0), shuffle(x_1, y_1)
    x, y = np.concatenate([x_0, x_1], axis=0), np.concatenate([y_0, y_1], axis=0)
    return x, y


def create_partitions(
    unpartitioned_dataset: XY,
    iid_fraction: float,
    num_partitions: int,
) -> XYList:
    """Create partitioned version of a training or test set.
    Currently tested and supported are MNIST, FashionMNIST and
    CIFAR-10/100
    """
    x, y = unpartitioned_dataset

    x, y = shuffle(x, y)
    x, y = sort_by_label_repeating(x, y)

    (x_0, y_0), (x_1, y_1) = split_at_fraction(x, y, fraction=iid_fraction)

    # Shift in second split of dataset the classes into two groups
    x_1, y_1 = shift(x_1, y_1)

    xy_0_partitions = partition(x_0, y_0, num_partitions)
    xy_1_partitions = partition(x_1, y_1, num_partitions)

    xy_partitions = combine_partitions(xy_0_partitions, xy_1_partitions)

    # Adjust x and y shape
    return [adjust_xy_shape(xy) for xy in xy_partitions]


def create_partitioned_dataset(
    keras_dataset: Tuple[XY, XY],
    iid_fraction: float,
    num_partitions: int,
) -> Tuple[PartitionedDataset, XY]:
    """Create partitioned version of keras dataset.
    Currently tested and supported are MNIST, FashionMNIST and
    CIFAR-10/100
    """
    xy_train, xy_test = keras_dataset

    xy_train_partitions = create_partitions(
        unpartitioned_dataset=xy_train,
        iid_fraction=iid_fraction,
        num_partitions=num_partitions,
    )

    xy_test_partitions = create_partitions(
        unpartitioned_dataset=xy_test,
        iid_fraction=iid_fraction,
        num_partitions=num_partitions,
    )

    return (xy_train_partitions, xy_test_partitions), adjust_xy_shape(xy_test)


def log_distribution(xy_partitions: XYList) -> None:
    """Print label distribution for list of paritions."""
    distro = [np.unique(y, return_counts=True) for _, y in xy_partitions]
    for d in distro:
        print(d)


def adjust_xy_shape(xy: XY) -> XY:
    """Adjust shape of both x and y."""
    x, y = xy
    if x.ndim == 3:
        x = adjust_x_shape(x)
    if y.ndim == 2:
        y = adjust_y_shape(y)
    return (x, y)


def adjust_x_shape(nda: np.ndarray) -> np.ndarray:
    """Turn shape (x, y, z) into (x, y, z, 1)."""
    nda_adjusted = np.reshape(nda, (nda.shape[0], nda.shape[1], nda.shape[2], 1))
    return nda_adjusted


def adjust_y_shape(nda: np.ndarray) -> np.ndarray:
    """Turn shape (x, 1) into (x)."""
    nda_adjusted = np.reshape(nda, (nda.shape[0]))
    return nda_adjusted


def split_array_at_indices(
    x: np.ndarray, split_idx: np.ndarray
) -> List[List[np.ndarray]]:
    """Splits an array `x` into list of elements using starting indices from
    `split_idx`.
        This function should be used with `unique_indices` from `np.unique()` after
        sorting by label.
    Args:
        x (np.ndarray): Original array of dimension (N,a,b,c,...)
        split_idx (np.ndarray): 1-D array contaning increasing number of
            indices to be used as partitions. Initial value must be zero. Last value
            must be less than N.
    Returns:
        List[List[np.ndarray]]: List of list of samples.
    """

    if split_idx.ndim != 1:
        raise ValueError("Variable `split_idx` must be a 1-D numpy array.")
    if split_idx.dtype != np.int64:
        raise ValueError("Variable `split_idx` must be of type np.int64.")
    if split_idx[0] != 0:
        raise ValueError("First value of `split_idx` must be 0.")
    if split_idx[-1] >= x.shape[0]:
        raise ValueError(
            """Last value in `split_idx` must be less than
            the number of samples in `x`."""
        )
    if not np.all(split_idx[:-1] <= split_idx[1:]):
        raise ValueError("Items in `split_idx` must be in increasing order.")

    num_splits: int = len(split_idx)
    split_idx = np.append(split_idx, x.shape[0])

    list_samples_split: List[List[np.ndarray]] = [[] for _ in range(num_splits)]
    for j in range(num_splits):
        tmp_x = x[split_idx[j] : split_idx[j + 1]]  # noqa: E203
        for sample in tmp_x:
            list_samples_split[j].append(sample)

    return list_samples_split


def exclude_classes_and_normalize(
    distribution: np.ndarray, exclude_dims: List[bool], eps: float = 1e-5
) -> np.ndarray:
    """Excludes classes from a distribution.
    This function is particularly useful when sampling without replacement.
    Classes for which no sample is available have their probabilities are set to 0.
    Classes that had probabilities originally set to 0 are incremented with
     `eps` to allow sampling from remaining items.
    Args:
        distribution (np.array): Distribution being used.
        exclude_dims (List[bool]): Dimensions to be excluded.
        eps (float, optional): Small value to be addad to non-excluded dimensions.
            Defaults to 1e-5.
    Returns:
        np.ndarray: Normalized distributions.
    """
    if np.any(distribution < 0) or (not np.isclose(np.sum(distribution), 1.0)):
        raise ValueError("distribution must sum to 1 and have only positive values.")

    if distribution.size != len(exclude_dims):
        raise ValueError(
            """Length of distribution must be equal
            to the length `exclude_dims`."""
        )
    if eps < 0:
        raise ValueError("""The value of `eps` must be positive and small.""")

    distribution[[not x for x in exclude_dims]] += eps
    distribution[exclude_dims] = 0.0
    sum_rows = np.sum(distribution) + np.finfo(float).eps
    distribution = distribution / sum_rows

    return distribution


def sample_without_replacement(
    distribution: np.ndarray,
    list_samples: List[List[np.ndarray]],
    num_samples: int,
    empty_classes: List[bool],
) -> Tuple[XY, List[bool]]:
    """Samples from a list without replacement using a given distribution.
    Args:
        distribution (np.ndarray): Distribution used for sampling.
        list_samples(List[List[np.ndarray]]): List of samples.
        num_samples (int): Total number of items to be sampled.
        empty_classes (List[bool]): List of booleans indicating which classes are empty.
            This is useful to differentiate which classes should still be sampled.
    Returns:
        XY: Dataset contaning samples
        List[bool]: empty_classes.
    """
    if np.sum([len(x) for x in list_samples]) < num_samples:
        raise ValueError(
            """Number of samples in `list_samples` is less than `num_samples`"""
        )

    # Make sure empty classes are not sampled
    # and solves for rare cases where
    if not empty_classes:
        empty_classes = len(distribution) * [False]

    distribution = exclude_classes_and_normalize(
        distribution=distribution, exclude_dims=empty_classes
    )

    data: List[np.ndarray] = []
    target: List[np.ndarray] = []

    for _ in range(num_samples):
        sample_class = np.where(np.random.multinomial(1, distribution) == 1)[0][0]
        sample: np.ndarray = list_samples[sample_class].pop()

        data.append(sample)
        target.append(sample_class)

        # If last sample of the class was drawn, then set the
        #  probability density function (PDF) to zero for that class.
        if len(list_samples[sample_class]) == 0:
            empty_classes[sample_class] = True
            # Be careful to distinguish between classes that had zero probability
            # and classes that are now empty
            distribution = exclude_classes_and_normalize(
                distribution=distribution, exclude_dims=empty_classes
            )
    data_array: np.ndarray = np.concatenate([data], axis=0)
    target_array: np.ndarray = np.array(target, dtype=np.int64)

    return (data_array, target_array), empty_classes


def get_partitions_distributions(partitions: XYList) -> Tuple[np.ndarray, List[int]]:
    """Evaluates the distribution over classes for a set of partitions.
    Args:
        partitions (XYList): Input partitions
    Returns:
        np.ndarray: Distributions of size (num_partitions, num_classes)
    """
    # Get largest available label
    labels = set()
    for _, y in partitions:
        labels.update(set(y))
    list_labels = sorted(list(labels))
    bin_edges = np.arange(len(list_labels) + 1)

    # Pre-allocate distributions
    distributions = np.zeros((len(partitions), len(list_labels)), dtype=np.float32)
    for idx, (_, _y) in enumerate(partitions):
        hist, _ = np.histogram(_y, bin_edges)
        distributions[idx] = hist / hist.sum()

    return distributions, list_labels


def create_lda_partitions(
    dataset: XY,
    dirichlet_dist: Optional[np.ndarray] = None,
    num_partitions: int = 100,
    concentration: Union[float, np.ndarray, List[float]] = 0.5,
    accept_imbalanced: bool = False,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> Tuple[XYList, np.ndarray]:
    """Create imbalanced non-iid partitions using Latent Dirichlet Allocation
    (LDA) without resampling.
    Args:
        dataset (XY): Dataset containing samples X and labels Y.
        dirichlet_dist (numpy.ndarray, optional): previously generated distribution to
            be used. This is useful when applying the same distribution for train and
            validation sets.
        num_partitions (int, optional): Number of partitions to be created.
            Defaults to 100.
        concentration (float, np.ndarray, List[float]): Dirichlet Concentration
            (:math:`\\alpha`) parameter. Set to float('inf') to get uniform partitions.
            An :math:`\\alpha \\to \\Inf` generates uniform distributions over classes.
            An :math:`\\alpha \\to 0.0` generates one class per client. Defaults to 0.5.
        accept_imbalanced (bool): Whether or not to accept imbalanced output classes.
            Default False.
        seed (None, int, SeedSequence, BitGenerator, Generator):
            A seed to initialize the BitGenerator for generating the Dirichlet
            distribution. This is defined in Numpy's official documentation as follows:
            If None, then fresh, unpredictable entropy will be pulled from the OS.
            One may also pass in a SeedSequence instance.
            Additionally, when passed a BitGenerator, it will be wrapped by Generator.
            If passed a Generator, it will be returned unaltered.
            See official Numpy Documentation for further details.
    Returns:
        Tuple[XYList, numpy.ndarray]: List of XYList containing partitions
            for each dataset and the dirichlet probability density functions.
    """
    # pylint: disable=too-many-arguments,too-many-locals

    x, y = dataset
    x, y = shuffle(x, y)
    x, y = sort_by_label(x, y)

    if (x.shape[0] % num_partitions) and (not accept_imbalanced):
        raise ValueError(
            """Total number of samples must be a multiple of `num_partitions`.
               If imbalanced classes are allowed, set
               `accept_imbalanced=True`."""
        )

    num_samples = num_partitions * [0]
    for j in range(x.shape[0]):
        num_samples[j % num_partitions] += 1

    # Get number of classes and verify if they matching with
    classes, start_indices = np.unique(y, return_index=True)

    # Make sure that concentration is np.array and
    # check if concentration is appropriate
    concentration = np.asarray(concentration)

    # Check if concentration is Inf, if so create uniform partitions
    partitions: List[XY] = [(_, _) for _ in range(num_partitions)]
    if float("inf") in concentration:

        partitions = create_partitions(
            unpartitioned_dataset=(x, y),
            iid_fraction=1.0,
            num_partitions=num_partitions,
        )
        dirichlet_dist = get_partitions_distributions(partitions)[0]

        return partitions, dirichlet_dist

    if concentration.size == 1:
        concentration = np.repeat(concentration, classes.size)
    elif concentration.size != classes.size:  # Sequence
        raise ValueError(
            f"The size of the provided concentration ({concentration.size}) ",
            f"must be either 1 or equal number of classes {classes.size})",
        )

    # Split into list of list of samples per class
    list_samples_per_class: List[List[np.ndarray]] = split_array_at_indices(
        x, start_indices
    )

    if dirichlet_dist is None:
        dirichlet_dist = np.random.default_rng(seed).dirichlet(
            alpha=concentration, size=num_partitions
        )

    if dirichlet_dist.size != 0:
        if dirichlet_dist.shape != (num_partitions, classes.size):
            raise ValueError(
                f"""The shape of the provided dirichlet distribution
                 ({dirichlet_dist.shape}) must match the provided number
                  of partitions and classes ({num_partitions},{classes.size})"""
            )

    # Assuming balanced distribution
    empty_classes = classes.size * [False]
    for partition_id in range(num_partitions):
        partitions[partition_id], empty_classes = sample_without_replacement(
            distribution=dirichlet_dist[partition_id].copy(),
            list_samples=list_samples_per_class,
            num_samples=num_samples[partition_id],
            empty_classes=empty_classes,
        )

    return partitions, dirichlet_dist


def construct_output_dir(args):
    path = Path("results")/args.dataset/f"alpha_{args.alpha}"/f"{args.base_model}_anyfed"/args.mode
    path = path/f"LE_{args.local_epoch}"/f"lr_{args.lr}"

    # now we append info about passed flags
    path = path/f"adpffn_{args.adpffn}"/f"anytime_{args.anytime}"/f"multi_tier_{args.multi_tier}"/f"freeze_base_{args.freeze_base}"/f"replace_{args.replace}"

    # append date and time
    path = path/str(datetime.now().strftime('%b%d_%H_%M_%S'))
    return path