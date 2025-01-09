import datetime
import os
from model import ML_BART, Sequence_Classifier, Token_Classifier
from transformers import BartConfig
import argparse
import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from dataset import load_pretrain
import copy
import random
from torch.optim import AdamW

pad = -1000

def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("--music_dim", type=int, default=128)
    parser.add_argument("--light_dim", type=int, nargs='+', default=[180,256])

    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--max_len', type=int, default=600)
    parser.add_argument('--gap', type=int, default=0)
    parser.add_argument('--heads', type=int, default=8)

    parser.add_argument('--attn_hs', type=int, default=128)

    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--converge_epoch', type=int, default=30)

    parser.add_argument('--data_path', type=str, default="./discard/test/data")
    parser.add_argument('--train_prop', type=float, default=0.5)


    args = parser.parse_args()
    return args


def iteration(data_loader,device,bart,model,optim,train=True):
    if train:
        torch.set_grad_enabled(True)
        bart.train()
        model.train()
    else:
        torch.set_grad_enabled(False)
        bart.eval()
        model.eval()

    acc_list = []

    pbar = tqdm.tqdm(data_loader, disable=False)
    for music, pos, neg in pbar:

        # 1. Process Music Emb
        music = music.float().to(device)
        # length = random.randint(0, 200)
        # music[:, 600 - length:, :] = pad
        non_pad = (music != pad).to(device)
        batch_size, seq_len, input_dim = music.shape
        rand_word = torch.randn((batch_size, seq_len, input_dim)).to(device)
        avg = torch.sum(music * non_pad, dim=1, keepdim=True) / (torch.sum(non_pad, dim=1, keepdim=True) + 1e-8)
        std = torch.sqrt(torch.sum(((music - avg) ** 2) * non_pad, dim=1, keepdim=True) / (
                torch.sum(non_pad, dim=1, keepdim=True) + 1e-8))
        rand_word = (rand_word + avg) * std
        music[~non_pad.bool()] = rand_word[~non_pad.bool()]
        attn_mask = non_pad[...,0].float()

        pos = pos.float().to(device)
        batch_size, seq_len, input_dim = pos.shape
        rand_word = torch.randn((batch_size, seq_len, input_dim)).to(device)
        pos_decoder = torch.zeros_like(pos)
        pos_decoder[:,1:,:] = pos[:,:-1,:]
        pos_decoder[:,0,:] = rand_word[:,0,:]
        pos = pos_decoder
        # length = random.randint(0, 200)
        # pos[:, 600 - length:, :] = pad
        non_pad = (pos != pad).to(device)
        avg = torch.sum(pos * non_pad, dim=1, keepdim=True) / (torch.sum(non_pad, dim=1, keepdim=True) + 1e-8)
        std = torch.sqrt(torch.sum(((pos - avg) ** 2) * non_pad, dim=1, keepdim=True) / (
                torch.sum(non_pad, dim=1, keepdim=True) + 1e-8))
        rand_word = (rand_word + avg) * std
        pos[~non_pad.bool()] = rand_word[~non_pad.bool()]
        attn_mask_pos = non_pad[..., 0].float()

        neg = neg.float().to(device)
        batch_size, seq_len, input_dim = neg.shape
        rand_word = torch.randn((batch_size, seq_len, input_dim)).to(device)
        neg_decoder = torch.zeros_like(neg)
        neg_decoder[:,1:,:] = neg[:,:-1,:]
        neg_decoder[:,0,:] = rand_word[:,0,:]
        neg = neg_decoder
        # length = random.randint(0, 200)
        # neg[:, 600 - length:, :] = pad
        non_pad = (neg != pad).to(device)
        avg = torch.sum(neg * non_pad, dim=1, keepdim=True) / (torch.sum(non_pad, dim=1, keepdim=True) + 1e-8)
        std = torch.sqrt(torch.sum(((neg - avg) ** 2) * non_pad, dim=1, keepdim=True) / (
                torch.sum(non_pad, dim=1, keepdim=True) + 1e-8))
        rand_word = (rand_word + avg) * std
        neg[~non_pad.bool()] = rand_word[~non_pad.bool()]
        attn_mask_neg = non_pad[..., 0].float()

        # 3. train
        y_pos = model(bart(music,pos,attn_mask,attn_mask_pos))
        y_neg = model(bart(music,neg,attn_mask,attn_mask_neg))
        gt_pos = torch.ones(batch_size, dtype=torch.long).to(device)
        gt_neg = torch.zeros(batch_size, dtype=torch.long).to(device)

        # y_pos = y_pos[:,0,:]
        # y_neg = y_neg[:,0,:]

        out_pos = torch.argmax(y_pos,dim=-1)
        out_neg = torch.argmax(y_neg,dim=-1)
        acc_pos = torch.mean((gt_pos == out_pos).float())
        acc_neg = torch.mean((gt_neg == out_neg).float())
        acc = (acc_neg + acc_pos) / 2
        acc_list.append(acc.item())

        # 4. calculate loss
        if train:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(y_pos, gt_pos) + loss_func(y_neg, gt_neg)
            optim.zero_grad()
            loss.backward()
            optim.step()

    return np.mean(acc_list)

def main():
    args = get_args()
    cuda_devices = args.cuda_devices
    if not args.cpu and cuda_devices is not None and len(cuda_devices) >= 1:
        device_name = "cuda:" + str(cuda_devices[0])
    else:
        device_name = "cpu"
    device = torch.device(device_name)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    date_str += '_MATCH'
    # mkdir results/{date_str}
    os.makedirs("results/{}".format(date_str), exist_ok=True)

    bartconfig = BartConfig(
        max_position_embeddings = args.max_len,
        encoder_layers = args.layers,
        encoder_ffn_dim = args.music_dim,
        encoder_attention_heads = args.heads,
        decoder_layers = args.layers,
        decoder_ffn_dim = args.music_dim,
        decoder_attention_heads = args.heads,
        d_model = args.music_dim
    )

    bart = ML_BART(bartconfig, class_num = args.light_dim, pretrain = True).to(device)
    model = Sequence_Classifier(class_num = 2, hs = args.music_dim, da = args.music_dim, r = args.heads).to(device)
    # model = Token_Classifier(hidden_dim = args.music_dim, class_num = 2).to(device)

    if len(cuda_devices) > 1 and not args.cpu:
        bart = nn.DataParallel(bart, device_ids=cuda_devices)
        model = nn.DataParallel(model, device_ids=cuda_devices)

    params = set(bart.parameters())| set(model.parameters())
    total_params = sum(p.numel() for p in params if p.requires_grad)
    print('total parameters:', total_params)
    optim = AdamW(params, lr=args.lr)#, weight_decay=0.01)

    best_acc = 0
    acc_epoch = 0
    j = 0

    train_data, test_data = load_pretrain(args.data_path, args.train_prop, args.max_len, args.gap)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=5)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=5)

    while True:
        j += 1
        acc = iteration(train_loader,device,bart,model,optim,train=True)
        log = "Epoch {} | Training Acc {:06f} | ".format(j, acc)
        print(log)
        with open("results/{}/log.txt".format(date_str), 'a') as file:
            file.write(log)
        acc = iteration(test_loader,device,bart,model,optim,train=False)
        log = "Testing Acc {:06f}".format(acc)
        print(log)
        with open("results/{}/log.txt".format(date_str), 'a') as file:
            file.write(log + "\n")
        if acc >= best_acc:
            torch.save(bart.state_dict(), "results/{}/bart_pretrain.pth".format(date_str))
            torch.save(model.state_dict(), "results/{}/head_pretrain.pth".format(date_str))
            best_acc = acc
            acc_epoch = 0
        else:
            acc_epoch += 1
        if acc_epoch >= args.converge_epoch:
            break
        print("Converge Epoch {:}".format(acc_epoch))


if __name__ == '__main__':
    import time
    start = time.time()
    main()
    end = time.time()
    print("Time:", time.strftime("%H:%M:%S", time.gmtime(end - start)))

    # python pretrain_match.py --data_path /mnt/disk/dian/m2l_data/output_mel/ --train_prop 0.9