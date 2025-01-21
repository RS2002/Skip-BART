from torch.utils.data import Dataset
import os
import pickle
import numpy as np
import random

pad = -1000
SELECTED_FEATURE = 'mel_spectrogram_db'
AUDIO_FOLDER = "output_mel"
LIGHT_FOLDER = "output"

min_db = -80.0
max_db = 0.0

class ML_Dataset(Dataset):
    def __init__(self, root_path, file_names, max_len = 600, gap = 0, fix_start = None):
        self.root_path = root_path
        self.file_names = file_names
        self.max_len = int(max_len)
        self.gap = int(gap)
        self.sample_len = self.max_len * (gap + 1)
        self.fix_start = fix_start

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        with open(os.path.join(self.root_path, AUDIO_FOLDER, self.file_names[index]), 'rb') as file:
            audio_data = pickle.load(file)
        with open(os.path.join(self.root_path, LIGHT_FOLDER, self.file_names[index]), 'rb') as file:
            light_data = pickle.load(file)
        music = audio_data[SELECTED_FEATURE].T
        music = music[::4, :]
        music_len = music.shape[0]
        music = (music - min_db) / (max_db - min_db)

        # use average value as light
        light = light_data['light']
        light_len = len(light)
        # align the length of light and music
        sample = np.linspace(0, light_len-1, music_len)
        h = [light[min(round(i), light_len-1)][60][0] for i in sample]
        v = [light[min(round(i), light_len-1)][0][1] for i in sample]
        h = np.array(h)
        v = np.array(v)
        # set the hue=0 as the average of the hue=1 and hue=179
        h[:,0] = (h[:,1]+h[:,179])//2
        h = np.argmax(h, axis=1)
        v[:,0] = 0
        values = np.arange(v.shape[1])
        weighted_sums = np.dot(v, values)
        counts = np.sum(v, axis=1)
        v = np.divide(weighted_sums, counts, out=np.zeros_like(weighted_sums, dtype=float), where=counts != 0)

        if music_len > self.sample_len:
            if self.fix_start:
                start_index = self.fix_start
            else:
                start_index = np.random.randint(0, music_len - self.sample_len + 1)
            music = music[start_index:start_index+self.sample_len, :]
            h = h[start_index:start_index+self.sample_len]
            v = v[start_index:start_index+self.sample_len]
        elif music_len < self.sample_len:
            music = np.concatenate([music, np.zeros([self.sample_len-music_len,music.shape[1]])+pad], axis=0)
            h = np.concatenate([h, np.zeros([self.sample_len-music_len])+pad], axis=0)
            v = np.concatenate([v, np.zeros([self.sample_len-music_len])+pad], axis=0)

        music = music[::(self.gap+1)]
        h = h[::(self.gap+1)]
        v = v[::(self.gap+1)]
        hv = np.stack([h, v], axis=1)

        f_name = self.file_names[index]
        return music, hv, f_name

class Pretrain_Dataset(Dataset):
    def __init__(self, file_path, max_len = 600, gap = 0):
        self.file_path = file_path
        self.max_len = int(max_len)
        self.gap = int(gap)
        self.sample_len = self.max_len * (gap + 1)
        self.num = len(self.file_path)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        with open(self.file_path[index], 'rb') as file:
            data = pickle.load(file)
        music_ori = data[SELECTED_FEATURE].T
        music_ori = music_ori[::4, :]
        music_len = music_ori.shape[0]
        music_ori = (music_ori - min_db) / (max_db - min_db)

        if music_len >= self.sample_len:
            start_index = np.random.randint(0, music_len - self.sample_len + 1)
            music = music_ori[start_index:start_index+self.sample_len, :]
        else:
            music = np.concatenate([music_ori, np.zeros([self.sample_len-music_len,music_ori.shape[1]])+pad], axis=0)
        music = music[::(self.gap+1)]

        if music_len >= self.sample_len:
            start_index = np.random.randint(0, music_len - self.sample_len + 1)
            pos = music_ori[start_index:start_index+self.sample_len, :]
        else:
            pos = np.concatenate([music_ori, np.zeros([self.sample_len-music_len,music_ori.shape[1]])+pad], axis=0)
        pos = pos[::(self.gap+1)]

        neg_index = random.randint(0, self.num - 1)
        while neg_index == index:
            neg_index = random.randint(0, self.num - 1)
        with open(self.file_path[neg_index], 'rb') as file:
            data = pickle.load(file)
        music_ori = data[SELECTED_FEATURE].T
        music_ori = music_ori[::4, :]
        music_len = music_ori.shape[0]            

        if music_len >= self.sample_len:
            start_index = np.random.randint(0, music_len - self.sample_len + 1)
            neg = music_ori[start_index:start_index+self.sample_len, :]
        else:
            neg = np.concatenate([music_ori, np.zeros([self.sample_len-music_len,music_ori.shape[1]])+pad], axis=0)
        neg = neg[::(self.gap+1)]

        return music, pos, neg

def load_data(root_path, train_prop = 0.9, max_len = 600, gap = 0, shuffle = False, random_seed = 42, fix_start = None):
    audio_path = os.path.join(root_path, AUDIO_FOLDER)
    # light_path = os.path.join(root_path, LIGHT_FOLDER)
    file_names = []
    for dirpath, _, filenames in os.walk(audio_path):
        for file in filenames:
            if file[-4:] == ".pkl":
                file_names.append(file)
    if train_prop == 1.0:
        return ML_Dataset(root_path, file_names, max_len, gap, fix_start)
    else:
        train_num = round(len(file_names) * train_prop)
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(file_names)
        return ML_Dataset(root_path,file_names[:train_num], max_len, gap, fix_start), ML_Dataset(root_path, file_names[train_num:], max_len, gap, fix_start)

def load_pretrain(root_path, train_prop = 0.9, max_len = 600, gap = 0):
    file_path = []
    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file[-4:] == ".pkl":
                file_path.append(os.path.join(dirpath, file))
    if train_prop == 1.0:
        return Pretrain_Dataset(file_path, max_len, gap)
    else:
        train_num = round(len(file_path) * train_prop)
        np.random.shuffle(file_path)
        return Pretrain_Dataset(file_path[:train_num], max_len, gap), Pretrain_Dataset(file_path[train_num:], max_len, gap)
