# python finetune.py --model_path bart_pretrain.pth --data_path ../m2l/output/ --train_prop 0.9 --cuda_devices 2

import datetime
import math
import os
from model import ML_BART, ML_Classifier
from transformers import BartConfig
import argparse
import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from dataset import load_data
import random
from torch.optim import AdamW

pad = -1000


def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("--music_dim", type=int, default=128)
    parser.add_argument("--light_dim", type=int, nargs='+', default=[180, 256])

    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--max_len', type=int, default=600)
    parser.add_argument('--gap', type=int, default=0)
    parser.add_argument('--heads', type=int, default=8)

    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0])
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--converge_epoch', type=int, default=30)
    parser.add_argument('--min_epoch', type=int, default=50)

    parser.add_argument('--data_path', type=str, default="./discard/test/data")
    parser.add_argument('--train_prop', type=float, default=0.9)

    parser.add_argument('--model_path', type=str, default=None)

    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument('--random_seed', type=int, default=42)

    args = parser.parse_args()
    return args


def iteration(data_loader, device, bart, model, optim, train=True, weight=[1.0, 1.0]):
    if train:
        torch.set_grad_enabled(True)
        bart.train()
        model.train()
    else:
        torch.set_grad_enabled(False)
        bart.eval()
        model.eval()

    h_acc_list = []
    v_acc_list = []
    loss_list = []

    pbar = tqdm.tqdm(data_loader, disable=False)
    for music, light, f_name in pbar:
        music = music.float().to(device)
        light = light.numpy()

        # # 0. Random Pad
        # length = random.randint(0, 200)
        # music[:, 600 - length:, :] = pad
        # light[:, 600 - length:, :] = pad

        # 1. Process Light
        light[light[..., 0] < 0, 0] = 0  # hue 0-179, [batch, seq_len]
        light[light[..., 1] < 0, 1] = 0  # value 0-255
        light = torch.from_numpy(light).float().to(device)  # [batch, seq_len, 2]
        light_sin = torch.sin(light[..., 0] * 2 * math.pi / 179)
        light_sin = light_sin.unsqueeze(-1) # [batch, seq_len, 1]
        light_cos = torch.cos(light[..., 0] * 2 * math.pi / 179)
        light_cos = light_cos.unsqueeze(-1) # [batch, seq_len, 1]
        light_hue = torch.cat([light_sin, light_cos], dim=-1) # [batch, seq_len, 2]
        light_value = light[..., 1] / 255 # [batch, seq_len]
        light_value = light_value.unsqueeze(-1) # [batch, seq_len, 1]
        light = torch.cat([light_hue, light_value], dim=-1) # [batch, seq_len, 3]
        
        light_input = torch.zeros_like(light)
        light_input[:, 1:, :] = light[:, :-1, :]
        light_input[:, 0, :] = light[:, 0, :]

        # 2. Process Music Emb
        non_pad = (music != pad).to(device)
        batch_size, seq_len, input_dim = music.shape
        rand_word = torch.randn((batch_size, seq_len, input_dim)).to(device)
        avg = torch.sum(music * non_pad, dim=1, keepdim=True) / (torch.sum(non_pad, dim=1, keepdim=True) + 1e-8)
        std = torch.sqrt(torch.sum(((music - avg) ** 2) * non_pad, dim=1, keepdim=True) / (
                torch.sum(non_pad, dim=1, keepdim=True) + 1e-8))
        rand_word = (rand_word + avg) * std
        rand_word = torch.clip(rand_word,0,1)
        music[~non_pad.bool()] = rand_word[~non_pad.bool()]
        attn_mask = non_pad[..., 0].float()
        attn_mask_light = torch.zeros_like(attn_mask)
        attn_mask_light[:, 1:] = attn_mask[:, :-1]
        attn_mask_light[:, 0] = attn_mask[:, 0]

        # 3. train
        h_hat, v_hat = model(bart(music, light_input, attn_mask, attn_mask_light))
        # h_out = torch.argmax(h_hat, dim=-1)
        # v_out = torch.argmax(v_hat, dim=-1)
        h_sin = h_hat[:, :, 0]
        h_cos = h_hat[:, :, 1]
        h_norm = torch.sqrt(h_sin ** 2 + h_cos ** 2)
        h_sin = h_sin / h_norm
        h_cos = h_cos / h_norm
        h_out = torch.stack([h_sin, h_cos], dim=-1) # [batch, seq_len, 2]
        v_out = v_hat # [batch, seq_len, 1]

        h_mse = torch.sum(((h_out - light[..., :2]) ** 2 * attn_mask.unsqueeze(-1)).sum(dim=-1)) / torch.sum(attn_mask)
        v_mse = torch.sum(((v_out - light[..., 2:]) ** 2 * attn_mask.unsqueeze(-1)).sum(dim=-1)) / torch.sum(attn_mask)
        h_acc_list.append(h_mse.item())
        v_acc_list.append(v_mse.item())

        # loss_func = nn.CrossEntropyLoss(reduction="none")
        loss_func = nn.MSELoss(reduction="none")
        h_out, v_out = h_out.reshape(batch_size * seq_len, -1), v_out.reshape(batch_size * seq_len, -1)
        h = light[..., :2].reshape(batch_size * seq_len, -1)  # [batch*seq_len, 2]
        v = light[..., 2].reshape(batch_size * seq_len, -1)   # [batch*seq_len, 1]
        attn_mask = attn_mask.reshape(batch_size * seq_len)
        loss_h = torch.sum(loss_func(h_out, h) * attn_mask.unsqueeze(-1)) / torch.sum(attn_mask)
        loss_v = torch.sum(loss_func(v_out, v) * attn_mask.unsqueeze(-1)) / torch.sum(attn_mask)
        loss = loss_h * weight[0] + loss_v * weight[1]
        loss_list.append((loss_h + loss_v).item())
        # 4. calculate loss
        if train:
            optim.zero_grad()
            loss.backward()
            optim.step()

    return np.mean(h_acc_list), np.mean(v_acc_list), np.mean(loss_list)


def main():
    args = get_args()
    cuda_devices = args.cuda_devices
    if not args.cpu and cuda_devices is not None and len(cuda_devices) >= 1:
        device_name = "cuda:" + str(cuda_devices[0])
    else:
        device_name = "cpu"
    device = torch.device(device_name)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    date_str += '_FINETUNE'
    # mkdir results/{date_str}
    os.makedirs("results/{}".format(date_str), exist_ok=True)

    bartconfig = BartConfig(
        max_position_embeddings=args.max_len,
        encoder_layers=args.layers,
        encoder_ffn_dim=args.music_dim,
        encoder_attention_heads=args.heads,
        decoder_layers=args.layers,
        decoder_ffn_dim=args.music_dim,
        decoder_attention_heads=args.heads,
        d_model=args.music_dim
    )

    bart = ML_BART(bartconfig).to(device)
    model = ML_Classifier(hidden_dim=args.music_dim).to(device)

    if len(cuda_devices) > 1 and not args.cpu:
        bart = nn.DataParallel(bart, device_ids=cuda_devices)
        model = nn.DataParallel(model, device_ids=cuda_devices)

    if args.model_path is not None:
        bart.load_state_dict(torch.load(args.model_path, weights_only=True))
        print("Load Model from ", args.model_path)
        bart.reset_decoder()
    else:
        print("No Pre-train Model")

    params = set(bart.parameters()) | set(model.parameters())
    total_params = sum(p.numel() for p in params if p.requires_grad)
    print('total parameters:', total_params)
    optim = AdamW(params, lr=args.lr)

    acc_best = 0
    acc_epoch = 0
    loss_best = 1e8
    loss_epoch = 0
    j = 0

    train_data, test_data = load_data(args.data_path, args.train_prop, args.max_len, args.gap, args.shuffle,
                                      args.random_seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=5)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=5)

    weight = [1.0, 1.0]
    while True:
        j += 1
        mse_h, mse_v, loss = iteration(train_loader, device, bart, model, optim, train=True, weight=weight)
        log = "Epoch {} | Training MSE_H {:.4f}, MSE_V {:.4f}, Loss {:.4f} | ".format(j, mse_h, mse_v, loss)
        print(log)
        with open("results/{}/log.txt".format(date_str), 'a') as file:
            file.write(log)
        mse_h, mse_v, loss = iteration(test_loader, device, bart, model, optim, train=False, weight=weight)
        log = "Test MSE_H {:.4f}, MSE_V {:.4f}, Loss {:.4f}".format(mse_h, mse_v, loss)
        print(log)
        with open("results/{}/log.txt".format(date_str), 'a') as file:
            file.write(log + "\n")
        mse_avg = (mse_h + mse_v) / 2
        weight = [(mse_v + 1e-8) / (mse_avg + 1e-8), (mse_h + 1e-8) / (mse_avg + 1e-8)]

        if j > args.min_epoch:
            if mse_avg <= mse_best or loss <= loss_best:
                torch.save(bart.state_dict(), "results/{}/bart_finetune.pth".format(date_str))
                torch.save(model.state_dict(), "results/{}/head_finetune.pth".format(date_str))

            if mse_avg <= mse_best:
                mse_best = mse_avg
                mse_epoch = 0
            else:
                mse_epoch += 1

            if loss <= loss_best:
                loss_best = loss
                loss_epoch = 0
            else:
                loss_epoch += 1

            if mse_epoch >= args.converge_epoch and loss_epoch > args.converge_epoch:
                break
        print("MSE Epoch {}, Loss Epoch {}".format(mse_epoch, loss_epoch))


if __name__ == '__main__':
    import time

    start = time.time()
    main()
    end = time.time()
    print("Time:", time.strftime("%H:%M:%S", time.gmtime(end - start)))

    # python finetune.py --train_prop 0.9 --cuda_devices 0 --data_path /mnt/disk/dian/m2l_data/ --model_path