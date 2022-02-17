# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import argparse
import copy
import glob
import os
import re
import time
import json
import warnings
from collections import defaultdict, OrderedDict
import matplotlib
from matplotlib import pyplot as plt

try:
    import nvidia_dlprof_pytorch_nvtx as pyprof
except ModuleNotFoundError:
    try:
        import pyprof
    except ModuleNotFoundError:
        warnings.warn('PyProf is unavailable')

import numpy as np
import torch
import torch.cuda.profiler as profiler
import torch.distributed as dist
import amp_C
from apex.optimizers import FusedAdam, FusedLAMB
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import common.tb_dllogger as logger
import models
from common.text import cmudict
from common.utils import prepare_tmp
from fastpitch.attn_loss_function import AttentionBinarizationLoss
from fastpitch.data_function import batch_to_gpu, TTSCollate, TTSDataset
from fastpitch.loss_function import FastPitchLoss


def parse_args(parser):
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('-d', '--dataset-path', type=str, default='./',
                        help='Path to dataset')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to a DLLogger log file')
    parser.add_argument('--pyprof', action='store_true',
                        help='Enable pyprof profiling')

    train = parser.add_argument_group('training setup')
    train.add_argument('--epochs', type=int, required=True,
                       help='Number of total epochs to run')
    train.add_argument('--epochs-per-checkpoint', type=int, default=50,
                       help='Number of epochs per checkpoint')
    train.add_argument('--checkpoint-path', type=str, default=None,
                       help='Checkpoint path to resume training')
    train.add_argument('--resume', action='store_true',
                       help='Resume training from the last checkpoint')
    train.add_argument('--seed', type=int, default=1234,
                       help='Seed for PyTorch random number generators')
    train.add_argument('--amp', action='store_true',
                       help='Enable AMP')
    train.add_argument('--cuda', action='store_true',
                       help='Run on GPU using CUDA')
    train.add_argument('--cudnn-benchmark', action='store_true',
                       help='Enable cudnn benchmark mode')
    train.add_argument('--ema-decay', type=float, default=0,
                       help='Discounting factor for training weights EMA')
    train.add_argument('--grad-accumulation', type=int, default=1,
                       help='Training steps to accumulate gradients for')
    train.add_argument('--kl-loss-start-epoch', type=int, default=250,
                       help='Start adding the hard attention loss term')
    train.add_argument('--kl-loss-warmup-epochs', type=int, default=100,
                       help='Gradually increase the hard attention loss term')
    train.add_argument('--kl-loss-weight', type=float, default=1.0,
                       help='Gradually increase the hard attention loss term')

    opt = parser.add_argument_group('optimization setup')
    opt.add_argument('--optimizer', type=str, default='lamb',
                     help='Optimization algorithm')
    opt.add_argument('-lr', '--learning-rate', type=float, required=True,
                     help='Learing rate')
    opt.add_argument('--weight-decay', default=1e-6, type=float,
                     help='Weight decay')
    opt.add_argument('--grad-clip-thresh', default=1000.0, type=float,
                     help='Clip threshold for gradients')
    opt.add_argument('-bs', '--batch-size', type=int, required=True,
                     help='Batch size per GPU')
    opt.add_argument('--warmup-steps', type=int, default=1000,
                     help='Number of steps for lr warmup')
    opt.add_argument('--dur-predictor-loss-scale', type=float,
                     default=1.0, help='Rescale duration predictor loss')
    opt.add_argument('--pitch-predictor-loss-scale', type=float,
                     default=1.0, help='Rescale pitch predictor loss')
    opt.add_argument('--attn-loss-scale', type=float,
                     default=1.0, help='Rescale alignment loss')

    data = parser.add_argument_group('dataset parameters')
    data.add_argument('--training-files', type=str, nargs='*', required=True,
                      help='Paths to training filelists.')
    data.add_argument('--validation-files', type=str, nargs='*',
                      required=True, help='Paths to validation filelists')
    data.add_argument('--text-cleaners', nargs='*',
                      default=['english_cleaners'], type=str,
                      help='Type of text cleaners for input text')
    data.add_argument('--symbol-set', type=str, default='olli',
                      help='Define symbol set for input text')
    data.add_argument('--p-arpabet', type=float, default=0.0,
                      help='Probability of using arpabets instead of graphemes '
                           'for each word; set 0 for pure grapheme training')
    data.add_argument('--heteronyms-path', type=str, default='cmudict/heteronyms',
                      help='Path to the list of heteronyms')
    data.add_argument('--cmudict-path', type=str, default='cmudict/cmudict',
                      help='Path to the pronouncing dictionary')
    data.add_argument('--prepend-space-to-text', action='store_true',
                      help='Capture leading silence with a space token')
    data.add_argument('--append-space-to-text', action='store_true',
                      help='Capture trailing silence with a space token')

    cond = parser.add_argument_group('data for conditioning')
    cond.add_argument('--n-speakers', type=int, default=1,
                      help='Number of speakers in the dataset. '
                           'n_speakers > 1 enables speaker embeddings')
    cond.add_argument('--load-pitch-from-disk', action='store_true',
                      help='Use pitch cached on disk with prepare_dataset.py')
    cond.add_argument('--pitch-online-method', default='pyin',
                      choices=['pyin'],
                      help='Calculate pitch on the fly during trainig')
    cond.add_argument('--pitch-online-dir', type=str, default=None,
                      help='A directory for storing pitch calculated on-line')
    cond.add_argument('--pitch-mean', type=float, default=214.72203,
                      help='Normalization value for pitch')
    cond.add_argument('--pitch-std', type=float, default=65.72038,
                      help='Normalization value for pitch')
    cond.add_argument('--load-mel-from-disk', action='store_true',
                      help='Use mel-spectrograms cache on the disk')  # XXX

    audio = parser.add_argument_group('audio parameters')
    audio.add_argument('--max-wav-value', default=32768.0, type=float,
                       help='Maximum audiowave value')
    audio.add_argument('--sampling-rate', default=22050, type=int,
                       help='Sampling rate')
    audio.add_argument('--filter-length', default=1024, type=int,
                       help='Filter length')
    audio.add_argument('--hop-length', default=256, type=int,
                       help='Hop (stride) length')
    audio.add_argument('--win-length', default=1024, type=int,
                       help='Window length')
    audio.add_argument('--mel-fmin', default=0.0, type=float,
                       help='Minimum mel frequency')
    audio.add_argument('--mel-fmax', default=8000.0, type=float,
                       help='Maximum mel frequency')

    dist = parser.add_argument_group('distributed setup')
    dist.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0),
                      help='Rank of the process for multiproc; do not set manually')
    dist.add_argument('--world_size', type=int, default=os.getenv('WORLD_SIZE', 1),
                      help='Number of processes for multiproc; do not set manually')
    return parser


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt.true_divide(num_gpus)


def init_distributed(args, world_size, rank):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing distributed training")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(backend=('nccl' if args.cuda else 'gloo'),
                            init_method='env://')
    print("Done initializing distributed training")


def last_checkpoint(output):

    def corrupted(fpath):
        try:
            torch.load(fpath, map_location='cpu')
            return False
        except:
            warnings.warn(f'Cannot load {fpath}')
            return True

    saved = sorted(
        glob.glob(f'{output}/FastPitch_checkpoint_*.pt'),
        key=lambda f: int(re.search('_(\d+).pt', f).group(1)))

    if len(saved) >= 1 and not corrupted(saved[-1]):
        return saved[-1]
    elif len(saved) >= 2:
        return saved[-2]
    else:
        return None


def maybe_save_checkpoint(args, model, ema_model, optimizer, scaler, epoch,
                          total_iter, config, final_checkpoint=False):
    if args.local_rank != 0:
        return

    intermediate = (args.epochs_per_checkpoint > 0
                    and epoch % args.epochs_per_checkpoint == 0)

    if not intermediate and epoch < args.epochs:
        return

    fpath = os.path.join(args.output, f"FastPitch_checkpoint_{epoch}.pt")
    print(f"Saving model and optimizer state at epoch {epoch} to {fpath}")
    ema_dict = None if ema_model is None else ema_model.state_dict()
    checkpoint = {'epoch': epoch,
                  'iteration': total_iter,
                  'config': config,
                  'state_dict': model.state_dict(),
                  'ema_state_dict': ema_dict,
                  'optimizer': optimizer.state_dict()}
    if args.amp:
        checkpoint['scaler'] = scaler.state_dict()
    torch.save(checkpoint, fpath)


def load_checkpoint(args, model, ema_model, optimizer, scaler, epoch,
                    total_iter, config, filepath):
    if args.local_rank == 0:
        print(f'Loading model and optimizer state from {filepath}')
    checkpoint = torch.load(filepath, map_location='cpu')
    epoch[0] = checkpoint['epoch'] + 1
    total_iter[0] = checkpoint['iteration']

    sd = {k.replace('module.', ''): v
          for k, v in checkpoint['state_dict'].items()}
    getattr(model, 'module', model).load_state_dict(sd)
    optimizer.load_state_dict(checkpoint['optimizer'])

    if args.amp:
        scaler.load_state_dict(checkpoint['scaler'])

    if ema_model is not None:
        ema_model.load_state_dict(checkpoint['ema_state_dict'])


with open(
      os.path.join('/data/tuong/Yen/MetaStyleSpeech/preprocessed_data/OLLI-SPEECH-1.7', "stats.json")
       ) as f:
    stats = json.load(f)
    stats = stats["pitch"] + stats["energy"][:2]


def validate(model, epoch, total_iter, criterion, valset, batch_size,
             collate_fn, distributed_run, batch_to_gpu, ema=False):
    """Handles all the validation scoring and printing"""
    was_training = model.training
    model.eval()

    tik = time.perf_counter()
 
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, num_workers=4, shuffle=False,
                                sampler=val_sampler,
                                batch_size=batch_size, pin_memory=False,
                                collate_fn=collate_fn)
        val_meta = defaultdict(float)
        val_num_frames = 0
        for i, batch in enumerate(val_loader):
            x, y, num_frames = batch_to_gpu(batch)
            y_pred = model(x)
            
            #mel_out, _, _, _, pitch_pred, pitch_tgt, energy_pred, energy_tgt, _, _, _, _ = [out[0] for out in y_pred] 
            #mel_tgt = y[0][0]
            #fig = plot_mel(
            #    [
            #        (mel_out.detach().cpu().numpy(), pitch_pred.detach().cpu().numpy(), energy_pred.detach().cpu().numpy()),
            #        (mel_tgt.cpu().numpy(), pitch_tgt.detach().cpu().numpy(), energy_tgt.detach().cpu().numpy()),
            #    ],
            #    stats,
            #    ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
            #)
            loss, meta = criterion(y_pred, y, is_training=False, meta_agg='sum')
            

            if distributed_run:
                for k, v in meta.items():
                    val_meta[k] += reduce_tensor(v, 1)
                val_num_frames += reduce_tensor(num_frames.data, 1).item()
            else:
                for k, v in meta.items():
                    val_meta[k] += v
                val_num_frames = num_frames.item()

        val_meta = {k: v / len(valset) for k, v in val_meta.items()}

    val_meta['took'] = time.perf_counter() - tik

    logger.log((epoch,) if epoch is not None else (),
               tb_total_steps=total_iter,
               subset='val_ema' if ema else 'val',
               data=OrderedDict([
                   ('loss', val_meta['loss'].item()),
                   ('mel_loss', val_meta['mel_loss'].item()),
                   ('frames/s', num_frames.item() / val_meta['took']),
                   ('took', val_meta['took'])
                   ]),
               )

    if was_training:
        model.train()
    return val_meta

def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = 90
    pitch_max = 600
    pitch_min = pitch_min
    pitch_max = pitch_max
    pitch_mean =  174.87988567095618
    pitch_std = 105.8709371279809

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    
    for i in range(len(data)):
        mel, pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato", linewidth=.7)
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet", linewidth=.7)
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


def adjust_learning_rate(total_iter, opt, learning_rate, warmup_iters=None):
    if warmup_iters == 0:
        scale = 1.0
    elif total_iter > warmup_iters:
        scale = 1. / (total_iter ** 0.5)
    else:
        scale = total_iter / (warmup_iters ** 1.5)

    for param_group in opt.param_groups:
        param_group['lr'] = learning_rate * scale


def apply_ema_decay(model, ema_model, decay):
    if not decay:
        return
    st = model.state_dict()
    add_module = hasattr(model, 'module') and not hasattr(ema_model, 'module')
    for k, v in ema_model.state_dict().items():
        if add_module and not k.startswith('module.'):
            k = 'module.' + k
        v.copy_(decay * v + (1 - decay) * st[k])


def init_multi_tensor_ema(model, ema_model):
    model_weights = list(model.state_dict().values())
    ema_model_weights = list(ema_model.state_dict().values())
    ema_overflow_buf = torch.cuda.IntTensor([0])
    return model_weights, ema_model_weights, ema_overflow_buf


def apply_multi_tensor_ema(decay, model_weights, ema_weights, overflow_buf):
    amp_C.multi_tensor_axpby(
        65536, overflow_buf, [ema_weights, model_weights, ema_weights],
        decay, 1-decay, -1)


def main():
    parser = argparse.ArgumentParser(description='PyTorch FastPitch Training',
                                     allow_abbrev=False)
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    if args.p_arpabet > 0.0:
        cmudict.initialize(args.cmudict_path, keep_ambiguous=True)

    distributed_run = args.world_size > 1

    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)

    if args.local_rank == 0:
        if not os.path.exists(args.output):
            os.makedirs(args.output)

    log_fpath = args.log_file or os.path.join(args.output, 'nvlog.json')
    tb_subsets = ['train', 'val']
    if args.ema_decay > 0.0:
        tb_subsets.append('val_ema')

    logger.init(log_fpath, args.output, enabled=(args.local_rank == 0),
                tb_subsets=tb_subsets)
    logger.parameters(vars(args), tb_subset='train')

    parser = models.parse_model_args('FastPitch', parser)
    args, unk_args = parser.parse_known_args()
    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if distributed_run:
        init_distributed(args, args.world_size, args.local_rank)

    device = torch.device('cuda' if args.cuda else 'cpu')
    model_config = models.get_model_config('FastPitch', args)
    model = models.get_model('FastPitch', model_config, device)

    attention_kl_loss = AttentionBinarizationLoss()

    # Store pitch mean/std as params to translate from Hz during inference
    model.pitch_mean[0] = args.pitch_mean
    model.pitch_std[0] = args.pitch_std

    kw = dict(lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9,
              weight_decay=args.weight_decay)
    if args.optimizer == 'adam':
        optimizer = FusedAdam(model.parameters(), **kw)
    elif args.optimizer == 'lamb':
        optimizer = FusedLAMB(model.parameters(), **kw)
    else:
        raise ValueError

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    if args.ema_decay > 0:
        ema_model = copy.deepcopy(model)
    else:
        ema_model = None

    if distributed_run:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True)

    if args.pyprof:
        pyprof.init(enable_function_stack=True)

    start_epoch = [1]
    start_iter = [0]

    assert args.checkpoint_path is None or args.resume is False, (
        "Specify a single checkpoint source")
    if args.checkpoint_path is not None:
        ch_fpath = args.checkpoint_path
    elif args.resume:
        ch_fpath = last_checkpoint(args.output)
    else:
        ch_fpath = None

    if ch_fpath is not None:
        load_checkpoint(args, model, ema_model, optimizer, scaler,
                        start_epoch, start_iter, model_config, ch_fpath)

    start_epoch = start_epoch[0]
    total_iter = start_iter[0]

    criterion = FastPitchLoss(
        dur_predictor_loss_scale=args.dur_predictor_loss_scale,
        pitch_predictor_loss_scale=args.pitch_predictor_loss_scale,
        attn_loss_scale=args.attn_loss_scale)

    collate_fn = TTSCollate()

    if args.local_rank == 0:
        prepare_tmp(args.pitch_online_dir)

    trainset = TTSDataset(audiopaths_and_text=args.training_files, **vars(args))
    valset = TTSDataset(audiopaths_and_text=args.validation_files, **vars(args))

    if distributed_run:
        train_sampler, shuffle = DistributedSampler(trainset), False
    else:
        train_sampler, shuffle = None, True

    # 4 workers are optimal on DGX-1 (from epoch 2 onwards)
    train_loader = DataLoader(trainset, num_workers=4, shuffle=shuffle,
                              sampler=train_sampler, batch_size=args.batch_size,
                              pin_memory=False, persistent_workers=False,
                              drop_last=True, collate_fn=collate_fn)

    if args.ema_decay:
        mt_ema_params = init_multi_tensor_ema(model, ema_model)

    model.train()

    if args.pyprof:
        torch.autograd.profiler.emit_nvtx().__enter__()
        profiler.start()

    epoch_loss = []
    epoch_mel_loss = []
    epoch_num_frames = []
    epoch_frames_per_sec = []
    epoch_time = []

    torch.cuda.synchronize()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.perf_counter()

        epoch_loss += [0.0]
        epoch_mel_loss += [0.0]
        epoch_num_frames += [0]
        epoch_frames_per_sec += [0.0]

        if distributed_run:
            train_loader.sampler.set_epoch(epoch)

        accumulated_steps = 0
        iter_loss = 0
        iter_num_frames = 0
        iter_meta = {}
        iter_start_time = None

        epoch_iter = 0
        num_iters = len(train_loader) // args.grad_accumulation
        for batch in train_loader:

            if accumulated_steps == 0:
                if epoch_iter == num_iters:
                    break
                total_iter += 1
                epoch_iter += 1
                if iter_start_time is None:
                    iter_start_time = time.perf_counter()

                adjust_learning_rate(total_iter, optimizer, args.learning_rate,
                                     args.warmup_steps)

                model.zero_grad(set_to_none=True)

            x, y, num_frames = batch_to_gpu(batch)

            with torch.cuda.amp.autocast(enabled=args.amp):
                y_pred = model(x)
                loss, meta = criterion(y_pred, y)
                if total_iter % 100 == 0 and args.local_rank==0:
                    mel_out, _, _, _, pitch_pred, pitch_tgt, energy_pred, energy_tgt, _, _, dur_tgt, _ = [out[0] for out in y_pred] 
                    mel_out, pitch_pred, pitch_tgt, energy_pred, energy_tgt, dur_tgt = mel_out.T.detach().cpu().numpy(), torch.squeeze(pitch_pred).detach().cpu().numpy(), torch.squeeze(pitch_tgt).detach().cpu().numpy(), energy_pred.detach().cpu().numpy(), energy_tgt.detach().cpu().numpy(), dur_tgt.detach().cpu().numpy() 
                    #pitch_pred = torch.squeeze(pitch_pred)
                    #pitch_tgt = torch.squeeze(pitch_tgt)
                    
                    mel_tgt, dur_len = y[0][0], y[0][1]
                    pitch_pred_full, pitch_tgt_full, energy_pred_full, energy_tgt_full = [], [], [], []

                    for i in range(dur_tgt.shape[0]):
                        pitch_pred_full += [pitch_pred[i]]*int(dur_tgt[i])
                        energy_pred_full += [energy_pred[i]]*int(dur_tgt[i])
                        pitch_tgt_full += [pitch_tgt[i]]*int(dur_tgt[i])
                        energy_tgt_full += [energy_tgt[i]]*int(dur_tgt[i])
                    #print(pitch_pred_full)
                    pitch_pred_full = np.asarray(pitch_pred_full)
                    pitch_tgt_full = np.asarray(pitch_tgt_full)
                    energy_pred_full = np.asarray(energy_pred_full)
                    energy_tgt_full = np.asarray(energy_tgt_full)

                    path = batch[-1][0].split('/')[-1]
                    fig = plot_mel(
                    [
                        (mel_out, pitch_pred_full, energy_pred_full),
                        (mel_tgt.cpu().numpy(), pitch_tgt_full, energy_tgt_full),
                    ],
                    stats,
                    ["Synthetized Spectrogram", "Ground-Truth Spectrogram"]
                )
                    logger.log(None, data={'tag': "step_{}_filename_{}".format(str(total_iter), path), 'fig': fig})
                if (args.kl_loss_start_epoch is not None
                        and epoch >= args.kl_loss_start_epoch):

                    if args.kl_loss_start_epoch == epoch and epoch_iter == 1:
                        print('Begin hard_attn loss')

                    _, _, _, _, _, _, _, _, attn_soft, attn_hard, _, _ = y_pred
                    binarization_loss = attention_kl_loss(attn_hard, attn_soft)
                    kl_weight = min((epoch - args.kl_loss_start_epoch) / args.kl_loss_warmup_epochs, 1.0) * args.kl_loss_weight
                    meta['kl_loss'] = binarization_loss.clone().detach() * kl_weight
                    loss += kl_weight * binarization_loss

                else:
                    meta['kl_loss'] = torch.zeros_like(loss)
                    kl_weight = 0
                    binarization_loss = 0

                loss /= args.grad_accumulation

            meta = {k: v / args.grad_accumulation
                    for k, v in meta.items()}

            if args.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if distributed_run:
                reduced_loss = reduce_tensor(loss.data, args.world_size).item()
                reduced_num_frames = reduce_tensor(num_frames.data, 1).item()
                meta = {k: reduce_tensor(v, args.world_size) for k, v in meta.items()}
            else:
                reduced_loss = loss.item()
                reduced_num_frames = num_frames.item()
            if np.isnan(reduced_loss):
                raise Exception("loss is NaN")

            accumulated_steps += 1
            iter_loss += reduced_loss
            iter_num_frames += reduced_num_frames
            iter_meta = {k: iter_meta.get(k, 0) + meta.get(k, 0) for k in meta}

            if accumulated_steps % args.grad_accumulation == 0:

                logger.log_grads_tb(total_iter, model)
                if args.amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_thresh)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_thresh)
                    optimizer.step()

                if args.ema_decay > 0.0:
                    apply_multi_tensor_ema(args.ema_decay, *mt_ema_params)

                iter_time = time.perf_counter() - iter_start_time
                iter_mel_loss = iter_meta['mel_loss'].item()
                iter_kl_loss = iter_meta['kl_loss'].item()
                epoch_frames_per_sec[-1] += iter_num_frames / iter_time
                epoch_loss[-1] += iter_loss
                epoch_num_frames[-1] += iter_num_frames
                epoch_mel_loss[-1] += iter_mel_loss

                logger.log((epoch, epoch_iter, num_iters),
                           tb_total_steps=total_iter,
                           subset='train',
                           data=OrderedDict([
                               ('loss', iter_loss),
                               ('mel_loss', iter_mel_loss),
                               ('kl_loss', iter_kl_loss),
                               ('kl_weight', kl_weight),
                               ('frames/s', iter_num_frames / iter_time),
                               ('took', iter_time),
                               ('lrate', optimizer.param_groups[0]['lr'])
                               ]),
                           )

                accumulated_steps = 0
                iter_loss = 0
                iter_num_frames = 0
                iter_meta = {}
                iter_start_time = time.perf_counter()



        # Finished epoch
        epoch_loss[-1] /= epoch_iter
        epoch_mel_loss[-1] /= epoch_iter
        epoch_time += [time.perf_counter() - epoch_start_time]
        iter_start_time = None

        logger.log((epoch,),
                   tb_total_steps=None,
                   subset='train_avg',
                   data=OrderedDict([
                       ('loss', epoch_loss[-1]),
                       ('mel_loss', epoch_mel_loss[-1]),
                       ('frames/s', epoch_num_frames[-1] / epoch_time[-1]),
                       ('took', epoch_time[-1])]),
                   )

        validate(model, epoch, total_iter, criterion, valset, args.batch_size,
                 collate_fn, distributed_run, batch_to_gpu)

        if args.ema_decay > 0:
            validate(ema_model, epoch, total_iter, criterion, valset,
                     args.batch_size, collate_fn, distributed_run, batch_to_gpu,
                     ema=True)

        maybe_save_checkpoint(args, model, ema_model, optimizer, scaler, epoch,
                              total_iter, model_config)
        logger.flush()

    # Finished training
    if args.pyprof:
        profiler.stop()
        torch.autograd.profiler.emit_nvtx().__exit__(None, None, None)

    if len(epoch_loss) > 0:
        # Was trained - average the last 20 measurements
        last_ = lambda l: np.asarray(l[-20:])
        epoch_loss = last_(epoch_loss)
        epoch_mel_loss = last_(epoch_mel_loss)
        epoch_num_frames = last_(epoch_num_frames)
        epoch_time = last_(epoch_time)
        logger.log((),
                   tb_total_steps=None,
                   subset='train_avg',
                   data=OrderedDict([
                       ('loss', epoch_loss.mean()),
                       ('mel_loss', epoch_mel_loss.mean()),
                       ('frames/s', epoch_num_frames.sum() / epoch_time.sum()),
                       ('took', epoch_time.mean())]),
                   )

    validate(model, None, total_iter, criterion, valset, args.batch_size,
             collate_fn, distributed_run, batch_to_gpu)


if __name__ == '__main__':
    main()