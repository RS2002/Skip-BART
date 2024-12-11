from model import ML_BART, Token_Classifier, Sequence_Classifier
from transformers import BartConfig, AdamW
import argparse
import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from dataset import load_data
import random
import copy
from sklearn.model_selection import train_test_split

pad = -1000

def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("--music_dim", type=int, default=512)
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

    parser.add_argument('--gan', action="store_true", default=False)

    args = parser.parse_args()
    return args


def iteration(data_loader,device,bart,model,discriminator,optim,optim_dis,train=True,gan=False):
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

        # 0. Random Pad
        # length = random.randint(0, 500)
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
                loss += gen_loss * 0.5

            optim.zero_grad()
            loss.backward()
            optim.step()

    return np.mean(mse_list)

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

    bart = ML_BART(bartconfig, class_num = args.light_dim, pretrain = True).to(device)
    model = Token_Classifier(hidden_dim = args.music_dim, class_num = args.music_dim).to(device)
    discriminator = Sequence_Classifier(class_num = 2, hs = args.music_dim, da = args.music_dim, r = args.heads).to(device)

    if len(cuda_devices) > 1 and not args.cpu:
        bart = nn.DataParallel(bart, device_ids=cuda_devices)
        model = nn.DataParallel(model, device_ids=cuda_devices)
        discriminator = nn.DataParallel(discriminator, device_ids=cuda_devices)

    params = set(bart.parameters())| set(model.parameters())
    total_params = sum(p.numel() for p in params if p.requires_grad)
    print('total parameters:', total_params)
    optim = AdamW(params, lr=args.lr)#, weight_decay=0.01)
    optim_dis = AdamW(discriminator.parameters(), lr=args.lr)#, weight_decay=0.01)

    best_mse = 1e8
    mse_epoch = 0
    j = 0

    dataset = load_data(args.data_path, 1.0, args.max_len, args.gap)
    train_data, test_data = train_test_split(dataset, test_size=1-args.train_prop)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=5)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=5)

    while True:
        j += 1
        mse = iteration(train_loader,device,bart,model,discriminator,optim,optim_dis,train=True,gan=True)
        log = "Epoch {} | Training MSE {:06f} | ".format(j, mse)
        print(log)
        with open("log.txt", 'a') as file:
            file.write(log)
        mse = iteration(test_loader,device,bart,model,discriminator,optim,optim_dis,train=False,gan=args.gan)
        log = "Testing MSE {:06f}".format(mse)
        print(log)
        with open("log.txt", 'a') as file:
            file.write(log + "\n")
        if mse <= best_mse:
            torch.save(bart.state_dict(), "bart_pretrain.pth")
            torch.save(model.state_dict(), "head_pretrain.pth")
            torch.save(discriminator.state_dict(), "discriminator.pth")
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
