from model import ML_BART, ML_Classifier
from transformers import BartConfig, AdamW
import argparse
import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from dataset import load_data
import random

pad = -1000

def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("--music_dim", type=int, default=512)
    parser.add_argument("--light_dim", type=int, nargs='+', default=[180,256])

    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--max_len', type=int, default=600)
    parser.add_argument('--gap', type=int, default=0)
    parser.add_argument('--heads', type=int, default=8)

    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--converge_epoch', type=int, default=30)

    parser.add_argument('--data_path', type=str, default="./discard/test/data")
    parser.add_argument('--train_prop', type=float, default=0.5)

    parser.add_argument('--model_path', type=str, default=None)

    parser.add_argument("--shuffle", action="store_true",default=False)
    parser.add_argument('--random_seed', type=int, default=42)


    args = parser.parse_args()
    return args


def iteration(data_loader,device,bart,model,optim,train=True,weight=[1.0,1.0]):
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

    pbar = tqdm.tqdm(data_loader, disable=False)
    for music, light, _ in pbar:
        music = music.float().to(device)
        light = light.numpy()
        rand_word = bart.music_mask

        # # 0. Random Pad
        # length = random.randint(0, 300)
        # music[:, 600 - length:, :] = pad
        # light[:, 600 - length:, :] = pad

        # 1. Tokenize Light
        light[light[..., 0] < 0, 0] = 180
        light[light[..., 1] < 0, 1] = 256

        light = torch.from_numpy(light)
        light = torch.round(light)
        light = light.long().to(device)

        light_input = torch.zeros_like(light)
        light_input[:,1:,:] = light[:,:-1,:]
        light_input[:,0,:] = light[:,0,:]

        # 2. Process Music Emb
        non_pad = (music != pad).to(device)
        batch_size, seq_len, input_dim = music.shape

        # rand_word = torch.randn((batch_size, seq_len, input_dim)).to(device)
        # avg = torch.sum(music * non_pad, dim=1, keepdim=True) / (torch.sum(non_pad, dim=1, keepdim=True) + 1e-8)
        # std = torch.sqrt(torch.sum(((music - avg) ** 2) * non_pad, dim=1, keepdim=True) / (
        #         torch.sum(non_pad, dim=1, keepdim=True) + 1e-8))
        # rand_word = (rand_word + avg) * std
        # music[~non_pad.bool()] = rand_word[~non_pad.bool()]

        attn_mask = non_pad[...,0].float()
        music[~attn_mask.bool()] = rand_word

        attn_mask_light = torch.zeros_like(attn_mask)
        attn_mask_light[:,1:] = attn_mask[:,:-1]
        attn_mask_light[:,0] = attn_mask[:,0]

        # 3. train
        h_hat, v_hat = model(bart(music,light_input,attn_mask,attn_mask_light))
        h_out = torch.argmax(h_hat,dim=-1)
        v_out = torch.argmax(v_hat,dim=-1)
        h_acc = torch.sum((h_out==light[...,0]).float() * attn_mask) / torch.sum(attn_mask)
        v_acc = torch.sum((v_out==light[...,1]).float() * attn_mask) / torch.sum(attn_mask)
        h_acc_list.append(h_acc.item())
        v_acc_list.append(v_acc.item())

        # 4. calculate loss
        if train:
            loss_func = nn.CrossEntropyLoss(reduction="none")
            h_hat, v_hat = h_hat.reshape(batch_size * seq_len, -1), v_hat.reshape(batch_size * seq_len, -1)
            h, v = light[...,0].reshape(batch_size * seq_len), light[...,1].reshape(batch_size * seq_len)
            attn_mask = attn_mask.reshape(batch_size * seq_len)
            loss_h = torch.sum(loss_func(h_hat,h)*attn_mask) / torch.sum(attn_mask)
            loss_v = torch.sum(loss_func(v_hat,v)*attn_mask) / torch.sum(attn_mask)
            loss = loss_h * weight[0] + loss_v * weight[1]
            optim.zero_grad()
            loss.backward()
            optim.step()

    return np.mean(h_acc_list), np.mean(v_acc_list)

def main():
    args = get_args()
    cuda_devices = args.cuda_devices
    if not args.cpu and cuda_devices is not None and len(cuda_devices) >= 1:
        device_name = "cuda:" + str(cuda_devices[0])
    else:
        device_name = "cpu"
    device = torch.device(device_name)

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

    bart = ML_BART(bartconfig, class_num = args.light_dim).to(device)
    model = ML_Classifier(hidden_dim = args.music_dim, class_num = args.light_dim).to(device)

    if len(cuda_devices) > 1 and not args.cpu:
        bart = nn.DataParallel(bart, device_ids=cuda_devices)
        model = nn.DataParallel(model, device_ids=cuda_devices)

    if args.model_path is not None:
        bart.load_state_dict(torch.load(args.model_path))
        bart.reset_decoder()
        print("Load Model from ", args.model_path)
    else:
        print("No Pre-train Model")

    params = set(bart.parameters())| set(model.parameters())
    total_params = sum(p.numel() for p in params if p.requires_grad)
    print('total parameters:', total_params)
    optim = AdamW(params, lr=args.lr, weight_decay=0.01)


    acc_best = 0
    acc_epoch = 0
    j = 0

    train_data, test_data = load_data(args.data_path, args.train_prop, args.max_len, args.gap, args.shuffle, args.random_seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=5)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=5)

    weight = [1.0, 1.0]
    while True:
        j += 1
        acc_h, acc_v = iteration(train_loader,device,bart,model,optim,train=True,weight=weight)
        log = "Epoch {} | Training Acc_H {:06f} , Acc_V {:06f} | ".format(j, acc_h, acc_v)
        print(log)
        with open("log.txt", 'a') as file:
            file.write(log)
        acc_h, acc_v = iteration(test_loader,device,bart,model,optim,train=False,weight=weight)
        log = "Test Acc_H {:06f} , Acc_V {:06f} ".format(acc_h, acc_v)
        print(log)
        with open("log.txt", 'a') as file:
            file.write(log + "\n")
        acc = (acc_h + acc_v) / 2
        weight = [ (acc_v + 1e-8) / (acc + 1e-8), (acc_h + 1e-8) / (acc + 1e-8)]
        if acc >= acc_best:
            torch.save(bart.state_dict(), "bart_finetune.pth")
            torch.save(model.state_dict(), "head_finetune.pth")
            acc_best = acc
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
