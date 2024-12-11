import pickle
from model import ML_BART, ML_Classifier
from transformers import BartConfig
import argparse
import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from dataset import load_data
from util import sampling

pad = -1000

def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("--music_dim", type=int, default=512)
    parser.add_argument("--light_dim", type=int, nargs='+', default=[180,256])

    parser.add_argument("--p", type=float, nargs='+', default=[0.9,0.9])
    parser.add_argument("--t", type=float, nargs='+', default=[1.1,1.1])

    parser.add_argument("--h_range", type=int, default=50)
    parser.add_argument("--v_range", type=int, nargs='+', default=[50,50])

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

    parser.add_argument('--bart_path', type=str, default="./bart_finetune.pth")
    parser.add_argument('--head_path', type=str, default="./head_finetune.pth")

    parser.add_argument("--shuffle", action="store_true",default=False)
    parser.add_argument('--random_seed', type=int, default=42)

    args = parser.parse_args()
    return args


def iteration(data_loader,device,bart,model,p,t,h_range=None,v_range=None):
    output = []

    pbar = tqdm.tqdm(data_loader, disable=False)
    for music, gt, f_name in pbar:
        music = music.float().to(device)
        gt = gt.float().to(device)
        light = torch.zeros_like(gt)
        light[...,0] += 180
        light[...,1] += 256
        light[:,0,:] = gt[:,0,:]
        light = torch.round(light)
        light = light.long()

        non_pad = (music != pad).to(device)
        attn_mask = non_pad[...,0].float()
        attn_mask_light = torch.zeros_like(attn_mask)
        attn_mask_light[:,1:] = attn_mask[:,:-1]
        attn_mask_light[:,0] = attn_mask[:,0]

        batch_size, seq_len, _ = music.shape
        result = torch.zeros([batch_size, seq_len, 2])
        result[:,0] = 180
        result[:,1] = 256

        for i in range(seq_len):
            h_temp, v_temp =  model(bart(music,light,attn_mask,attn_mask_light))

            for j in range(batch_size):
                if attn_mask[j,i] == 1:
                    h_last, v_last = light[j,i,0], light[j,i,1]
                    if v_last != 256 and v_range is not None:
                        v_left = max(0, v_last - v_range[0])
                        v_right = min(255, v_last + v_range[1])
                        v_temp[i,j,:v_left] = 1e-8
                        v_temp[i,j,v_right:] = 1e-8
                    if h_last != 180 and h_range is not None:
                        h_left = h_last - h_range
                        h_right = h_last + h_range

                        if h_left>=0 and h_right<=179:
                            h_temp[i, j, :h_left] = 1e-8
                            h_temp[i, j, h_right:] = 1e-8
                        elif h_left<0 and h_right<=179:
                            h_left = 180 + h_left
                            if h_left < h_right:
                                h_temp[i, j, h_left:h_right] = 1e-8
                        elif h_left>=0 and h_right>179:
                            h_right = h_right - 179
                            if h_right < h_left:
                                h_temp[i, j, h_right:h_left] = 1e-8

                    h = sampling(h_temp[j,i,:-1],p=p[0],t=t[0])
                    v = sampling(v_temp[j,i,:-1],p=p[1],t=t[1])

                    result[j,i,0], result[j,i,1] = h, v
                    if i != seq_len - 1:
                        light[j, i + 1, 0], light[j, i + 1, 1] = h, v

        output.append(result.cpu().detach())

    output = torch.cat(output, dim=0)
    return output

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

    bart.load_state_dict(torch.load(args.bart_path))
    model.load_state_dict(torch.load(args.head_path))

    torch.set_grad_enabled(False)
    bart.eval()
    model.eval()

    _, test_data = load_data(args.data_path, args.train_prop, args.max_len, args.gap, args.shuffle, args.random_seed, fix_start=0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=5)
    ground_truth = []
    f_names = []
    for i in range(len(test_data)):
        music, hv, f_name = test_data[i]
        ground_truth.append(hv)
        f_names.append(f_name)
    ground_truth = np.stack(ground_truth, axis=0)
    output = iteration(test_loader,device,bart,model,args.p,args.t)
    output = output.numpy()
    res = {
        'ground_truth': ground_truth,
        'output': output,
        'f_names': f_names
    }
    with open('light_pred.pkl', 'wb') as f:
        pickle.dump(res, f)


if __name__ == '__main__':
    main()