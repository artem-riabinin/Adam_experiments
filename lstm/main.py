import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import copy
import wandb

# wandb logging
wandb_log = True 
wandb_project = 'lstm_exp'
wandb_run_name = 'fullbatch_beta1_0_beta2_0.999_lr_0.001'
if wandb_log:
    run = wandb.init(project=wandb_project, name=wandb_run_name)

import data
import model

import utils
from utils import batchify, get_batch, repackage_hidden, get_model_grads, get_model_params, norm_diff, CSVLogger

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=512,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0,
                    help='gradient clipping')
parser.add_argument('--beta1', type=float, default=0,
                    help='beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='beta2')
parser.add_argument('--epochs', type=int, default=500,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=2640, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.0,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.0,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.0,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.0,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='report interval')
parser.add_argument('--smooth-log-interval', type=int, default=1, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=0,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='adam',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
args = parser.parse_args()
args.tied = True
args.save = 'ckpts/' + args.save
# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 1
corpus.train = corpus.train[:-5589]
train_data = batchify(corpus.train, args.batch_size, args)
eval_data = train_data
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss
criterion = None

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
###
if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop
        for rnn in model.rnns:
            if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
            elif rnn.zoneout > 0: rnn.zoneout = args.wdrop
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
###
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)

def eval_grad(model):
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    model.zero_grad()
#     optimizer.zero_grad()
    while i < train_data.size(0)/10:
        #bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        bptt = args.bptt
        # Prevent excessively small or negative sequence lengths
        #seq_len = max(5, int(np.random.normal(bptt, 5)))
        seq_len = bptt
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)
        total_norm = 0
        for p in model.parameters():
            if p is None or p.grad is None:
                continue
            param_normsq = p.grad.data.norm(2)**2
            total_norm += param_normsq.item()
        
        loss = raw_loss + args.wdecay * total_norm
        # Activiation Regularization
        if args.alpha!=0: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta!=0: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
#         if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
#         optimizer.step()

#         total_loss += raw_loss.data
        if batch % args.log_interval == 0 and batch > 0:
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len
        
    gradnorm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        gradnorm += param_norm.item() ** 2
    gradnorm = gradnorm ** (1. / 2) / batch
    
    return gradnorm


def train():
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    accumulation_steps = train_data.size(0) // args.bptt
    while i < train_data.size(0):
        prev_model = copy.deepcopy(model)
        #bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        bptt = args.bptt
        # Prevent excessively small or negative sequence lengths
        #seq_len = max(5, int(np.random.normal(bptt, 5)))
        seq_len = bptt
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)
        raw_loss = raw_loss / accumulation_steps

        loss = raw_loss
        # Activiation Regularization
        if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:]) 
        
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip != 0: torch.nn.utils.clip_grad_norm_(params, args.clip)
        
        total_loss += raw_loss.data
        
        if (batch + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            cur_loss = total_loss.item()
            print(f"| epoch {epoch:3d} | total loss {cur_loss:.2f} |")
            if wandb_log:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": cur_loss,
                })
        
        ###
        batch += 1
        i += seq_len
    return cur_loss



def eval_smooth(prev_model, model, num_pts=1):
    alphas = np.arange(1, num_pts+1)/(num_pts+1)
    gnorm = eval_grad(prev_model)
    update_size = utils.norm_diff(utils.get_model_params(model), \
                                  utils.get_model_params(prev_model))
    max_smooth = -1
    for alpha in alphas:
        new_model = copy.deepcopy(prev_model)
        
        for n, p in new_model.named_parameters():
            p.data = alpha * p.data + (1-alpha) * {n:p for n, p in model.named_parameters()}[n].data
            
        eval_grad(new_model)
        smooth = utils.norm_diff(utils.get_model_grads(new_model), utils.get_model_grads(prev_model))/ (update_size * (1- alpha))
        max_smooth = max(smooth, max_smooth)
    
    return max_smooth, gnorm


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000
train_loss_lst, grad_norm_lst, val_loss_lst = [], [], []
# smooth_lst = []
csv_path = os.path.join(args.save, 'epoch.csv')
csv_logger_keys = ['train_loss', 'valid_loss', 'grad_norm']
csvlogger = CSVLogger(csv_path, args, csv_logger_keys)

csv_path = os.path.join(args.save, 'iteration.csv')
csv_logger_keys = []
csv_logger_keys.extend(['smoothness', 'grad_norm'])
iterationlogger = CSVLogger(csv_path, args, csv_logger_keys)

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=0)#args.wdecay)
    if args.optimizer == 'adam':
        betas = (args.beta1, args.beta2)
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=betas, weight_decay=args.wdecay)
        
        
    
    
#     prev_model = copy.deepcopy(model)
    
    
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        
        grad_norm = eval_grad(model)
        train_loss = train()

            
        prev_model = copy.deepcopy(model)

        
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | smoothness {:5.2f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), smooth_lst[-1]))
            print('-' * 89)

            if val_loss2 < stored_loss:
                model_save(args.save+'.pt')
                print('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss = evaluate(val_data, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | grad norm {:5.2f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), grad_norm))
            print('-' * 89)

            if val_loss < stored_loss:
                model_save(args.save+'.pt')
                print('Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Reduce lr due to validation plateau')
                optimizer.param_groups[0]['lr'] /= 2.

#                 optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when:
                print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save+'.pt', epoch))
                print('Dividing learning rate by 10')
            
            
#             optimizer.param_groups[0]['lr'] *= (epoch/(epoch+1))**0.5

            best_val_loss.append(val_loss)
            
        train_loss_lst.append(train_loss)
        val_loss_lst.append(val_loss)
        grad_norm_lst.append(grad_norm)
        csvlogger.write_row([train_loss, val_loss, grad_norm])

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(args.save+'.pt')

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)

run.finish()
