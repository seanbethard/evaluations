#!/usr/bin/python3

# apt install python3.10-venv ipython3 -y
# pip install einops transformers mosaicml matplotlib

import time
import torch.utils.data
torch.manual_seed(413)

from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import composer
from composer.loggers import InMemoryLogger
from composer.trainer import Trainer
from composer.optim import DecoupledSGDW, LinearWithWarmupScheduler, ConstantScheduler
from composer.algorithms import LabelSmoothing, BlurPool, ProgressiveResizing
from composer import functional as cf
from composer import models


# task: CIFAR-10, image classification
# objectives:
#   train a baseline ResNet model with decoupled weight decay regularization
#   add smoothing, a low-pass filter and progressive regularization in a pre-training phase to the baseline
#   pre-training downscale the images, increase throughput and image size until images are default scale
#   extend the baseline training phase by one epoch


# compute
device = "gpu"

# prepare datasets
data_directory = "./data"


# batch size
batch_size = 1024

# define an epoch
one_epoch = '1ep'

# specify max training duration
max_epoch_count = 11

baseline_epochs = str(max_epoch_count-1)+"ep"
baseline_plotname = baseline_epochs+"-baseline"
pr_plot = baseline_epochs+"-pr"


baseline_model = models.composer_resnet_cifar(model_name='resnet_56', num_classes=10)

# decouple weight decay from learning rate
baseline_optimizer = DecoupledSGDW(
    baseline_model.parameters(),
    lr=0.05,
    momentum=0.9,
    weight_decay=2.0e-3
)

# the baseline lr is flat
baseline_scheduler = LinearWithWarmupScheduler(
    t_warmup=one_epoch,
    alpha_i=1.0,
    alpha_f=1.0
)

# normalization constants
mean, std = (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
cifar10_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

train_dataset = datasets.CIFAR10(data_directory, train=True, download=True, transform=cifar10_transforms)
test_dataset = datasets.CIFAR10(data_directory, train=False, download=True, transform=cifar10_transforms)

# prepare training data
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

baseline_logger = InMemoryLogger()
trainer = Trainer(
    loggers=baseline_logger,
    device=device,
    max_duration=baseline_epochs,
    model=baseline_model,
    optimizers=baseline_optimizer,
    schedulers=baseline_scheduler,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
)

start_time = time.perf_counter()
trainer.fit()
trainer.close()
end_time = time.perf_counter()
baseline_time = end_time - start_time
print(f"Baseline SGDW completed in {baseline_time:0.4f}")

# plot baseline accuracy
timeseries_raw = baseline_logger.get_timeseries("metrics/eval/MulticlassAccuracy")
plt.plot(timeseries_raw['epoch'], timeseries_raw["metrics/eval/MulticlassAccuracy"])
plt.xlabel("Epochs")
plt.ylabel("Validation accuracy")
plt.title("Baseline accuracy per epoch")
plt.savefig(baseline_plotname)

# close the baseline trainer


# allocate time to a pre-training phase and downsize the training data
# here pre-training is 2/3 of the total training time
# during pre-training
#   linearly resize the training data back to default scale
#   increase throughput to avoid picking up unwanted features from downsized images
smooth_JJ_labels_NNS = cf.smooth_labels(
    logits=torch.logits,
    target=test_dataloader,
    smoothing=0.1)

composer_algorithms = [smooth_JJ_labels_NNS,
                       BlurPool(replace_convs=True, replace_maxpools=True, blur_first=True),
                       ProgressiveResizing(finetune_fraction=0.34, initial_scale=.6)]


# train the same model with smoothing, a low-pass filter and progressive resizing
pr_model = models.composer_resnet_cifar(model_name="resnet_56", num_classes=10)
pr_logger = InMemoryLogger()
trainer = Trainer(
    loggers=pr_logger,
    device=device,
    max_duration=baseline_epochs,
    algorithms=composer_algorithms,
    model=pr_model,
    optimizers=baseline_optimizer,
    schedulers=baseline_scheduler,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
)

start_time = time.perf_counter()
trainer.fit()
trainer.close()
end_time = time.perf_counter()
pr_time = end_time - start_time
print(f"Progressive resizing completed in {pr_time:0.4f} seconds")

no_pr_logger = InMemoryLogger()
trainer = Trainer(
    model=pr_model,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    max_duration=one_epoch,
    optimizers=baseline_optimizer,
    schedulers=ConstantScheduler(alpha=1.0, t_max='1dur'),
    device=device,
    loggers=no_pr_logger,
    algorithms=[LabelSmoothing(0.1)]
)

start_time = time.perf_counter()
trainer.fit()
trainer.close()
end_time = time.perf_counter()
no_pr_accelerated_time = end_time - start_time

no_pr_time = pr_time + no_pr_accelerated_time
print(f"Training with resizing and no resize epoch completed in {no_pr_time:0.4f}")


baseline_timeseries = baseline_logger.get_timeseries("metrics/eval/MulticlassAccuracy")
baseline_epoch = baseline_timeseries['epoch']
baseline_acc = baseline_timeseries["metrics/eval/MulticlassAccuracy"]

with_algorithms_timeseries = pr_logger.get_timeseries("metrics/eval/MulticlassAccuracy")
with_algorithms_epochs = list(with_algorithms_timeseries["epoch"])
with_algorithms_acc = list(with_algorithms_timeseries["metrics/eval/MulticlassAccuracy"])

bonus_epoch_timeseries = no_pr_logger.get_timeseries("metrics/eval/MulticlassAccuracy")
bonus_epoch_epochs = [with_algorithms_epochs[-1] + i for i in bonus_epoch_timeseries["epoch"]]
with_algorithms_epochs.extend(bonus_epoch_epochs)
with_algorithms_acc.extend(bonus_epoch_timeseries["metrics/eval/MulticlassAccuracy"])

print("Baseline Validation Mean: " + str(sum(baseline_acc)/len(baseline_acc)))
print("Progressive Resizing Validation Mean: " + str(sum(with_algorithms_acc)/len(with_algorithms_acc)))

plt.plot(baseline_epoch, baseline_acc, label="Baseline SGDW")
plt.plot(with_algorithms_epochs, with_algorithms_acc, label="Progressive resizing")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Validation accuracy")
plt.title("Validation accuracy with equivalent wall clock time")
plt.savefig(pr_plot)


"""
******************************
Config:
composer_commit_hash: None
composer_version: 0.17.0
node_name: unknown because NODENAME environment variable not set
num_gpus_per_node: 1
num_nodes: 1
rank_zero_seed: 3638092877

******************************
  warnings.warn(f'Cannot split tensor of length {len(t)} into batches of size {microbatch_size}.')
It took 1227.4740 seconds to train
"""


"""
******************************
Config:
blurpool/num_blurconv_layers: 4
blurpool/num_blurpool_layers: 0
composer_commit_hash: None
composer_version: 0.17.0
enabled_algorithms/BlurPool: true
enabled_algorithms/LabelSmoothing: true
enabled_algorithms/ProgressiveResizing: true
node_name: unknown because NODENAME environment variable not set
num_gpus_per_node: 1
num_nodes: 1
rank_zero_seed: 1681769880

******************************
  warnings.warn('Some targets have less than 1 total probability.')
  warnings.warn(f'Cannot split tensor of length {len(t)} into batches of size {microbatch_size}.')
  It took 1068.3773 seconds to train
"""


"""
******************************
Config:
composer_commit_hash: None
composer_version: 0.17.0
enabled_algorithms/LabelSmoothing: true
node_name: unknown because NODENAME environment variable not set
num_gpus_per_node: 1
num_nodes: 1
rank_zero_seed: 2005466561

******************************
It took 1078.1140 seconds to train
"""

"""
Baseline validation mean: 0.7156601594761014
Validation mean with resizing: 0.5867736441220424
"""