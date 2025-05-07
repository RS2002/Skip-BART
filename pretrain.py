import datetime
import os
from model import ML_BART, Sequence_Classifier, Token_Predictor
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
from peft import get_peft_model, LoraConfig

pad = -1000

def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("--music_dim", type=int, default=512)
    parser.add_argument("--light_dim", type=int, nargs='+', default=[180,256])
    parser.add_argument('--gap', type=int, default=0)

    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=1024)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--hs', type=int, default=1024)
    parser.add_argument('--ffn_dims', type=int, default=2048)

    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0])
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--converge_epoch', type=int, default=30)
    parser.add_argument('--min_epoch', type=int, default=50)

    parser.add_argument('--data_path', type=str, default="./discard/test/data")
    parser.add_argument('--train_prop', type=float, default=0.9)

    parser.add_argument("--encoder_only", action="store_true",default=False)


    args = parser.parse_args()
    return args

def iteration(data_loader,device,bart,model_mlm,model_match,discriminator,optim,optim_dis,train=True,gan=True,encoder_only=False):
    if train:
        torch.set_grad_enabled(True)
        bart.train()
        model_mlm.train()
        model_match.train()
        discriminator.train()
    else:
        torch.set_grad_enabled(False)
        bart.eval()
        model_mlm.eval()
        model_match.eval()
        discriminator.eval()

    mse_list = []
    acc_list = []
    pbar = tqdm.tqdm(data_loader, disable=False)
    for music, pos, neg in pbar:
        # rand = random.random()
        rand = 0.0
        if rand < 0.5:
            music = music.float().to(device)

            # # 0. Random Pad
            # length = random.randint(0, 200)
            # music[:, 600 - length:, :] = pad

            # 1. Process Music Emb
            non_pad = (music != pad).to(device)
            batch_size, seq_len, input_dim = music.shape
            rand_word = torch.randn((batch_size, seq_len, input_dim)).to(device)
            avg = torch.sum(music * non_pad, dim=1, keepdim=True) / (torch.sum(non_pad, dim=1, keepdim=True) + 1e-8)
            std = torch.sqrt(torch.sum(((music - avg) ** 2) * non_pad, dim=1, keepdim=True) / (
                    torch.sum(non_pad, dim=1, keepdim=True) + 1e-8))
            rand_word = (rand_word + avg) * std
            music[~non_pad.bool()] = rand_word[~non_pad.bool()]
            attn_mask = non_pad[..., 0].float()

            music_decoder = torch.zeros_like(music)
            music_decoder[:, 1:, :] = music[:, :-1, :]
            music_decoder[:, 0, :] = rand_word[:, 0, :]
            attn_mask_decoder = torch.zeros_like(attn_mask)
            attn_mask_decoder[:, 1:] = attn_mask[:, :-1]
            attn_mask_decoder[:, 0] = 0

            # 2. Random MASK
            loss_mask = torch.zeros([batch_size, seq_len]).to(device)
            chosen_num_min = int(seq_len * 0.15)
            chosen_num_max = int(seq_len * 0.30)
            num_ones = torch.randint(chosen_num_min, chosen_num_max + 1, (batch_size,))
            row_indices = torch.arange(batch_size).unsqueeze(1).repeat(1, chosen_num_max)
            col_indices = torch.randint(0, seq_len, (batch_size, chosen_num_max))
            loss_mask[row_indices[:, :num_ones.max()], col_indices[:, :num_ones.max()]] = 1
            loss_mask[~non_pad[..., 0]] = 0
            input = copy.deepcopy(music)
            input[loss_mask.bool()] = rand_word[loss_mask.bool()]

            # 3. train
            if encoder_only:
                music_hat = model_mlm(bart.encode(input, attn_mask))
            else:
                music_hat = model_mlm(bart(input, music_decoder, attn_mask, attn_mask_decoder))

            loss_mse = nn.MSELoss(reduction="none")
            loss_mask = loss_mask.unsqueeze(2).repeat(1, 1, input_dim)
            loss1 = torch.sum(loss_mse(music_hat, music) * loss_mask) / torch.sum(loss_mask)
            loss2 = torch.sum(loss_mse(music_hat, music) * non_pad) / torch.sum(non_pad)
            loss = loss1 * 0.8 + loss2 * 0.2
            mse_list.append(loss1.item())

            # 4. calculate loss
            if train:
                if gan:
                    loss_cls = nn.CrossEntropyLoss()
                    non_pad = non_pad.float()
                    music_hat = music_hat * non_pad + rand_word * (1 - non_pad)
                    truth_hat = discriminator(music)
                    false_hat = discriminator(music_hat.detach())
                    false = torch.zeros(batch_size, dtype=torch.long).to(device)
                    truth = torch.ones(batch_size, dtype=torch.long).to(device)
                    loss_truth = loss_cls(truth_hat, truth)
                    loss_false = loss_cls(false_hat, false)
                    dis_loss = loss_truth + loss_false

                    optim_dis.zero_grad()
                    dis_loss.backward()
                    optim_dis.step()

                    gen_loss = loss_cls(discriminator(music_hat), truth)
                    loss += gen_loss * 0.1

                optim.zero_grad()
                loss.backward()
                optim.step()
        else:
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
            attn_mask = non_pad[..., 0].float()

            loss_mask = torch.zeros([batch_size, seq_len]).to(device)
            chosen_num_min = int(seq_len * 0.10)
            chosen_num_max = int(seq_len * 0.50)
            num_ones = torch.randint(chosen_num_min, chosen_num_max + 1, (batch_size,))
            row_indices = torch.arange(batch_size).unsqueeze(1).repeat(1, chosen_num_max)
            col_indices = torch.randint(0, seq_len, (batch_size, chosen_num_max))
            loss_mask[row_indices[:, :num_ones.max()], col_indices[:, :num_ones.max()]] = 1
            loss_mask[~non_pad[..., 0]] = 0
            music[loss_mask.bool()] = rand_word[loss_mask.bool()]

            pos = pos.float().to(device)
            batch_size, seq_len, input_dim = pos.shape
            rand_word = torch.randn((batch_size, seq_len, input_dim)).to(device)
            pos_decoder = torch.zeros_like(pos)
            pos_decoder[:, 1:, :] = pos[:, :-1, :]
            pos_decoder[:, 0, :] = rand_word[:, 0, :]
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

            loss_mask = torch.zeros([batch_size, seq_len]).to(device)
            chosen_num_min = int(seq_len * 0.10)
            chosen_num_max = int(seq_len * 0.50)
            num_ones = torch.randint(chosen_num_min, chosen_num_max + 1, (batch_size,))
            row_indices = torch.arange(batch_size).unsqueeze(1).repeat(1, chosen_num_max)
            col_indices = torch.randint(0, seq_len, (batch_size, chosen_num_max))
            loss_mask[row_indices[:, :num_ones.max()], col_indices[:, :num_ones.max()]] = 1
            loss_mask[~non_pad[..., 0]] = 0
            pos[loss_mask.bool()] = rand_word[loss_mask.bool()]

            neg = neg.float().to(device)
            batch_size, seq_len, input_dim = neg.shape
            rand_word = torch.randn((batch_size, seq_len, input_dim)).to(device)
            neg_decoder = torch.zeros_like(neg)
            neg_decoder[:, 1:, :] = neg[:, :-1, :]
            neg_decoder[:, 0, :] = rand_word[:, 0, :]
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

            loss_mask = torch.zeros([batch_size, seq_len]).to(device)
            chosen_num_min = int(seq_len * 0.10)
            chosen_num_max = int(seq_len * 0.50)
            num_ones = torch.randint(chosen_num_min, chosen_num_max + 1, (batch_size,))
            row_indices = torch.arange(batch_size).unsqueeze(1).repeat(1, chosen_num_max)
            col_indices = torch.randint(0, seq_len, (batch_size, chosen_num_max))
            loss_mask[row_indices[:, :num_ones.max()], col_indices[:, :num_ones.max()]] = 1
            loss_mask[~non_pad[..., 0]] = 0
            neg[loss_mask.bool()] = rand_word[loss_mask.bool()]

            # 2. train
            y_pos = model_match(bart(music, pos, attn_mask, attn_mask_pos))
            y_neg = model_match(bart(music, neg, attn_mask, attn_mask_neg))
            gt_pos = torch.ones(batch_size, dtype=torch.long).to(device)
            gt_neg = torch.zeros(batch_size, dtype=torch.long).to(device)
            out_pos = torch.argmax(y_pos, dim=-1)
            out_neg = torch.argmax(y_neg, dim=-1)
            acc_pos = torch.mean((gt_pos == out_pos).float())
            acc_neg = torch.mean((gt_neg == out_neg).float())
            acc = (acc_neg + acc_pos) / 2
            acc_list.append(acc.item())

            # 3. calculate loss
            if train:
                loss_func = nn.CrossEntropyLoss()
                loss = loss_func(y_pos, gt_pos) + loss_func(y_neg, gt_neg)
                optim.zero_grad()
                loss.backward()
                optim.step()
    if len(mse_list) == 0:
        mse_list.append(1e8)
    if len(acc_list) == 0:
        acc_list.append(0)
    return np.mean(mse_list), np.mean(acc_list)




def iteration_mlm(data_loader,device,bart,model,discriminator,optim,optim_dis,train=True,gan=True):
    if train:
        torch.set_grad_enabled(True)
        bart.train()
        model.train()
        discriminator.train()
    else:
        torch.set_grad_enabled(False)
        bart.eval()
        model.eval()
        discriminator.eval()

    mse_list = []

    pbar = tqdm.tqdm(data_loader, disable=False)
    for music, _, _ in pbar:
        music = music.float().to(device)

        # # 0. Random Pad
        # length = random.randint(0, 200)
        # music[:, 600 - length:, :] = pad

        # 1. Process Music Emb
        non_pad = (music != pad).to(device)
        batch_size, seq_len, input_dim = music.shape
        rand_word = torch.randn((batch_size, seq_len, input_dim)).to(device)
        avg = torch.sum(music * non_pad, dim=1, keepdim=True) / (torch.sum(non_pad, dim=1, keepdim=True) + 1e-8)
        std = torch.sqrt(torch.sum(((music - avg) ** 2) * non_pad, dim=1, keepdim=True) / (
                torch.sum(non_pad, dim=1, keepdim=True) + 1e-8))
        rand_word = (rand_word + avg) * std
        music[~non_pad.bool()] = rand_word[~non_pad.bool()]
        attn_mask = non_pad[...,0].float()

        music_decoder = torch.zeros_like(music)
        music_decoder[:,1:,:] = music[:,:-1,:]
        music_decoder[:,0,:] = rand_word[:,0,:]
        attn_mask_decoder = torch.zeros_like(attn_mask)
        attn_mask_decoder[:,1:] = attn_mask[:,:-1]
        attn_mask_decoder[:,0] = 0

        # 2. Random MASK
        loss_mask = torch.zeros([batch_size, seq_len]).to(device)
        chosen_num_min = int(seq_len * 0.15)
        chosen_num_max = int(seq_len * 0.30)
        num_ones = torch.randint(chosen_num_min, chosen_num_max + 1, (batch_size,))
        row_indices = torch.arange(batch_size).unsqueeze(1).repeat(1, chosen_num_max)
        col_indices = torch.randint(0, seq_len, (batch_size, chosen_num_max))
        loss_mask[row_indices[:, :num_ones.max()], col_indices[:, :num_ones.max()]] = 1
        loss_mask[~non_pad[..., 0]] = 0
        input = copy.deepcopy(music)
        input[loss_mask.bool()] = rand_word[loss_mask.bool()]

        # 3. train
        if encoder_only:
            music_hat = model(bart.encode(input,attn_mask))
        else:
            music_hat = model(bart(input,music_decoder,attn_mask,attn_mask_decoder))

        loss_mse = nn.MSELoss(reduction="none")
        loss_mask = loss_mask.unsqueeze(2).repeat(1, 1, input_dim)
        loss1 = torch.sum(loss_mse(music_hat, music) * loss_mask) / torch.sum(loss_mask)
        loss2 = torch.sum(loss_mse(music_hat, music) * non_pad) / torch.sum(non_pad)
        loss = loss1 * 0.8 + loss2 * 0.2
        mse_list.append(loss1.item())


        # 4. calculate loss
        if train:
            if gan:
                loss_cls = nn.CrossEntropyLoss()
                non_pad = non_pad.float()
                music_hat = music_hat * non_pad + rand_word * (1 - non_pad)
                truth_hat = discriminator(music)
                false_hat = discriminator(music_hat.detach())
                false = torch.zeros(batch_size, dtype=torch.long).to(device)
                truth = torch.ones(batch_size, dtype=torch.long).to(device)
                loss_truth = loss_cls(truth_hat, truth)
                loss_false = loss_cls(false_hat, false)
                dis_loss = loss_truth + loss_false

                optim_dis.zero_grad()
                dis_loss.backward()
                optim_dis.step()

                gen_loss = loss_cls(discriminator(music_hat), truth)
                loss += gen_loss * 0.1

            optim.zero_grad()
            loss.backward()
            optim.step()

    return np.mean(mse_list)

def iteration_match(data_loader,device,bart,model,optim,train=True):
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

        loss_mask = torch.zeros([batch_size, seq_len]).to(device)
        chosen_num_min = int(seq_len * 0.10)
        chosen_num_max = int(seq_len * 0.50)
        num_ones = torch.randint(chosen_num_min, chosen_num_max + 1, (batch_size,))
        row_indices = torch.arange(batch_size).unsqueeze(1).repeat(1, chosen_num_max)
        col_indices = torch.randint(0, seq_len, (batch_size, chosen_num_max))
        loss_mask[row_indices[:, :num_ones.max()], col_indices[:, :num_ones.max()]] = 1
        loss_mask[~non_pad[..., 0]] = 0
        music[loss_mask.bool()] = rand_word[loss_mask.bool()]



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

        loss_mask = torch.zeros([batch_size, seq_len]).to(device)
        chosen_num_min = int(seq_len * 0.10)
        chosen_num_max = int(seq_len * 0.50)
        num_ones = torch.randint(chosen_num_min, chosen_num_max + 1, (batch_size,))
        row_indices = torch.arange(batch_size).unsqueeze(1).repeat(1, chosen_num_max)
        col_indices = torch.randint(0, seq_len, (batch_size, chosen_num_max))
        loss_mask[row_indices[:, :num_ones.max()], col_indices[:, :num_ones.max()]] = 1
        loss_mask[~non_pad[..., 0]] = 0
        pos[loss_mask.bool()] = rand_word[loss_mask.bool()]



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

        loss_mask = torch.zeros([batch_size, seq_len]).to(device)
        chosen_num_min = int(seq_len * 0.10)
        chosen_num_max = int(seq_len * 0.50)
        num_ones = torch.randint(chosen_num_min, chosen_num_max + 1, (batch_size,))
        row_indices = torch.arange(batch_size).unsqueeze(1).repeat(1, chosen_num_max)
        col_indices = torch.randint(0, seq_len, (batch_size, chosen_num_max))
        loss_mask[row_indices[:, :num_ones.max()], col_indices[:, :num_ones.max()]] = 1
        loss_mask[~non_pad[..., 0]] = 0
        neg[loss_mask.bool()] = rand_word[loss_mask.bool()]



        # 2. train
        y_pos = model(bart(music,pos,attn_mask,attn_mask_pos))
        y_neg = model(bart(music,neg,attn_mask,attn_mask_neg))
        gt_pos = torch.ones(batch_size, dtype=torch.long).to(device)
        gt_neg = torch.zeros(batch_size, dtype=torch.long).to(device)
        out_pos = torch.argmax(y_pos,dim=-1)
        out_neg = torch.argmax(y_neg,dim=-1)
        acc_pos = torch.mean((gt_pos == out_pos).float())
        acc_neg = torch.mean((gt_neg == out_neg).float())
        acc = (acc_neg + acc_pos) / 2
        acc_list.append(acc.item())

        # 3. calculate loss
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
    date_str += '_PRETRAIN'
    # mkdir results/{date_str}
    os.makedirs("results/{}".format(date_str), exist_ok=True)

    # bartconfig = BartConfig(
    #     max_position_embeddings = args.max_len,
    #     encoder_layers = args.layers,
    #     encoder_ffn_dim = args.music_dim,
    #     encoder_attention_heads = args.heads,
    #     decoder_layers = args.layers,
    #     decoder_ffn_dim = args.music_dim,
    #     decoder_attention_heads = args.heads,
    #     d_model = args.music_dim
    # )

    bartconfig = BartConfig(max_position_embeddings=args.max_len,
                               d_model=args.hs,
                               encoder_layers=args.layers,
                               encoder_ffn_dim=args.ffn_dims,
                               encoder_attention_heads=args.heads,
                               decoder_layers=args.layers,
                               decoder_ffn_dim=args.ffn_dims,
                               decoder_attention_heads=args.heads
                               )

    bart = ML_BART(bartconfig, class_num = args.light_dim, pretrain = True).to(device)
    model = Sequence_Classifier(class_num = 2, hs = args.hs, da = args.hs, r = args.heads).to(device)
    predictor = Token_Predictor(hidden_dim=args.hs, class_num=args.music_dim).to(device)
    discriminator = Sequence_Classifier(class_num=2, hs=args.music_dim, da=args.music_dim, r=args.heads).to(device)
    

    bart.load_state_dict(torch.load("./pianobart.pth"),strict=False)
    bart.bart = get_peft_model(bart.bart, bart.lora_config)


    if len(cuda_devices) > 1 and not args.cpu:
        bart = nn.DataParallel(bart, device_ids=cuda_devices)
        model = nn.DataParallel(model, device_ids=cuda_devices)
        predictor = nn.DataParallel(predictor, device_ids=cuda_devices)
        discriminator = nn.DataParallel(discriminator, device_ids=cuda_devices)

    params = set(bart.parameters())| set(model.parameters()) | set(predictor.parameters())
    total_params = sum(p.numel() for p in params if p.requires_grad)
    print('total parameters:', total_params)
    optim = AdamW(params, lr=args.lr)
    optim_dis = AdamW(discriminator.parameters(), lr=args.lr)

    best_mse = 1e8
    mse_epoch = 0
    best_acc = 0
    acc_epoch = 0
    j = 0

    train_data, test_data = load_pretrain(args.data_path, args.train_prop, args.max_len, args.gap)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=5)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=5)

    while True:
        j += 1

        mse, acc = iteration(train_loader,device,bart,predictor,model,discriminator,optim,optim_dis,train=True,gan=True,encoder_only=args.encoder_only)
        # log = "Epoch {} | Training MSE {:06f} , Training Acc {:06f} | ".format(j, mse,acc)
        log = "Epoch {} | Training MSE {:06f} ".format(j, mse)
        print(log)
        with open("results/{}/log.txt".format(date_str), 'a') as file:
            file.write(log)

        # mse = iteration_mlm(train_loader, device, bart, predictor, discriminator, optim, optim_dis, train=True, gan=True)
        # log = "Epoch {} | Training MSE {:06f} , ".format(j, mse)
        # print(log)
        # with open("results/{}/log.txt".format(date_str), 'a') as file:
        #     file.write(log)
        #
        # acc = iteration_match(train_loader,device,bart,model,optim,train=True)
        # log = "Training Acc {:06f} | ".format(acc)
        # print(log)
        # with open("results/{}/log.txt".format(date_str), 'a') as file:
        #     file.write(log)

        mse = iteration_mlm(test_loader, device, bart, predictor, discriminator, optim, optim_dis, train=False, gan=False,encoder_only=args.encoder_only)
        log = "Testing MSE {:06f} , ".format(mse)
        print(log)
        with open("results/{}/log.txt".format(date_str), 'a') as file:
            # file.write(log)
            file.write(log + "\n")


        # acc = iteration_match(test_loader,device,bart,model,optim,train=False)
        # log = "Testing Acc {:06f}".format(acc)
        # print(log)
        # with open("results/{}/log.txt".format(date_str), 'a') as file:
        #     file.write(log + "\n")


        # if acc >= best_acc or mse <= best_mse:
        #     torch.save(bart.state_dict(), "results/{}/bart_pretrain.pth".format(date_str))
        # if acc > best_acc:
        #     best_acc = acc
        #     acc_epoch = 0
        # else:
        #     acc_epoch += 1
        # if mse < best_mse:
        #     best_mse = mse
        #     mse_epoch = 0
        # else:
        #     mse_epoch += 1
        # if acc_epoch >= args.converge_epoch and mse_epoch >= args.converge_epoch:
        #     break
        # print("Acc Epoch {:}, MSE Epoch {:}".format(acc_epoch,mse_epoch))

        if j > args.min_epoch:
            if mse <= best_mse:
                torch.save(bart.state_dict(), "results/{}/bart_pretrain.pth".format(date_str))
            if mse < best_mse:
                best_mse = mse
                mse_epoch = 0
            else:
                mse_epoch += 1
            if mse_epoch >= args.converge_epoch:
                break
        print("Converge Epoch {:}".format(mse_epoch))



if __name__ == '__main__':
    import time
    start = time.time()
    main()
    end = time.time()
    print("Time:", time.strftime("%H:%M:%S", time.gmtime(end - start)))