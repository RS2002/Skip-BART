from model import ML_BART, ML_Classifier
from transformers import BartConfig
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from util import sampling
from peft import get_peft_model, LoraConfig
import openl3
import librosa
import os
import cv2
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.editor import VideoFileClip
import random

pad = -1000

def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("--music_dim", type=int, default=512)
    parser.add_argument("--light_dim", type=int, nargs='+', default=[180,256])

    parser.add_argument("--p", type=float, nargs='+', default=[0.9,0.9])
    parser.add_argument("--t", type=float, nargs='+', default=[1.1,1.1])

    parser.add_argument("--h_range", type=int, default=50)
    parser.add_argument("--v_range", type=int, nargs='+', default=[50,50])

    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=1024)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--hs', type=int, default=1024)
    parser.add_argument('--ffn_dims', type=int, default=2048)

    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0])

    parser.add_argument('--music_file', type=str, default="test.wav")
    parser.add_argument('--emb_file', type=str, default=None)
    parser.add_argument('--gen_file', type=str, default=None)

    parser.add_argument('--bart_path', type=str, default="/home/dian/Code/ML-BART/zzj/LoRA/results/2025-02-07-19-51-58_FINETUNE/bart_finetune.pth")
    parser.add_argument('--head_path', type=str, default="/home/dian/Code/ML-BART/zzj/LoRA/results/2025-02-07-19-51-58_FINETUNE/head_finetune.pth")

    args = parser.parse_args()
    return args

def iteration(music,device,bart,model,p,t,h_range=None,v_range=None):

    music = music.float().to(device)
    light = torch.zeros([music.shape[0],music.shape[1],2]).to(device)
    light[...,0] += 180
    light[...,1] += 256
    light[:,0,0] = random.randint(0, 180)
    light[:,0,1] = random.randint(0, 255)

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

    return result.cpu().detach()


def generate_video(h, v, save_path, audio_path, target_length=None):
    # 确保临时目录存在
    os.makedirs('no_upload/res', exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    audio = AudioFileClip(audio_path)
    fps = 10
    out = cv2.VideoWriter('no_upload/res/temp.mp4', fourcc, fps, (400, 400))

    # 如果指定了目标长度，则调整数据长度
    if target_length is not None and len(h) != target_length:
        if len(h) < target_length:
            # 如果数据长度小于目标长度，则重复最后一帧
            h = np.pad(h, (0, target_length - len(h)), mode='edge')
            v = np.pad(v, (0, target_length - len(v)), mode='edge')
        else:
            # 如果数据长度大于目标长度，则截断
            h = h[:target_length]
            v = v[:target_length]

    for frame in tqdm(range(len(h))):
        true_img_hv = np.ones((400, 400, 3), dtype=np.uint8) * 255

        true_img_hv[:, :, 0] = h[frame]
        true_img_hv[:, :, 1] = 255
        true_img_hv[:, :, 2] = v[frame]

        true_img_bgr_hv = cv2.cvtColor(true_img_hv, cv2.COLOR_HSV2BGR)
        combined = true_img_bgr_hv

        out.write(combined)

    out.release()

    video = VideoFileClip('no_upload/res/temp.mp4')

    video_duration = len(h) / fps

    if audio.duration > video_duration:
        audio = audio.subclip(0, video_duration)
    else:
        # 如果音频比视频短，则循环音频
        repeats = int(np.ceil(video_duration / audio.duration))
        if repeats > 1:
            audio = audio.loop(repeats)
            audio = audio.subclip(0, video_duration)

    final_video = video.set_audio(audio)
    final_video.write_videofile(save_path, codec='libx264')

    video.close()
    audio.close()

    return len(h)


def gen(predictions,audio_path,save_path,max_length=1024):

    pred_h = predictions[:, 0]
    pred_h = pred_h[pred_h != -1000]
    pred_v = predictions[:, 1]
    pred_v = pred_v[pred_v != -1000]

    target_length = min(max_length, len(pred_h))

    generate_video(pred_h, pred_v, save_path, audio_path, target_length=target_length)

if __name__ == '__main__':
    args = get_args()

    # load file

    music_file = args.music_file
    emb_file = args.emb_file
    gen_file = args.gen_file
    input_name = music_file.split(".")[-2]

    if gen_file is not None:
        output = np.load(gen_file)
    else:
        if emb_file is not None:
            embeddings = np.load(emb_file)
        else:
            audio, sr = librosa.load(music_file, sr=None)
            embeddings, timestamps = openl3.get_audio_embedding(
                audio,
                sr,
                embedding_size=512
            )
            np.save(f'{input_name}_emb.npy', embeddings)

        if embeddings.shape[0]>args.max_len:
            music = embeddings[:args.max_len,:]
        else:
            music = embeddings.copy()

        # load model
        cuda_devices = args.cuda_devices
        if not args.cpu and cuda_devices is not None and len(cuda_devices) >= 1:
            device_name = "cuda:" + str(cuda_devices[0])
        else:
            device_name = "cpu"
        device = torch.device(device_name)

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

        bart = ML_BART(bartconfig, class_num=args.light_dim).to(device)
        model = ML_Classifier(hidden_dim=args.hs, class_num=args.light_dim).to(device)

        bart.bart = get_peft_model(bart.bart, bart.lora_config)

        if len(cuda_devices) > 1 and not args.cpu:
            bart = nn.DataParallel(bart, device_ids=cuda_devices)
            model = nn.DataParallel(model, device_ids=cuda_devices)

        bart.load_state_dict(torch.load(args.bart_path), strict=True)
        model.load_state_dict(torch.load(args.head_path), strict=True)

        torch.set_grad_enabled(False)
        bart.eval()
        model.eval()

        music = torch.from_numpy(music).unsqueeze(0).to(device)
        output = iteration(music,device,bart,model,args.p,args.t)
        output = output.squeeze(0).cpu().detach().numpy()
        np.save(f'{input_name}_gen.npy', embeddings)

    # print(output.shape)
    gen(output, music_file, f'{input_name}.mp4', max_length=1024)