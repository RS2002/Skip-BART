from torch.utils.data import Dataset
import os
import pickle
import numpy as np

pad = -1000

class ML_Dataset(Dataset):
    def __init__(self, file_path, max_len = 600, gap = 0):
        self.file_path = file_path
        self.max_len = int(max_len)
        self.gap = int(gap)
        self.sample_len = self.max_len * (gap + 1)

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, index):
        with open(self.file_path[index], 'rb') as file:
            data = pickle.load(file)
        music = data['music']
        music_len = music.shape[0]

        # use average value as light
        sample = data['sample']
        light = data['light']

        h = [light[int(i)][0][2][1:] for i in sample]
        h = np.array(h)
        values = np.arange(h.shape[1])
        weighted_sums = np.dot(h, values)
        counts = np.sum(h, axis=1)
        h = np.divide(weighted_sums, counts, out=np.zeros_like(weighted_sums, dtype=float), where=counts != 0)

        if music_len > self.sample_len:
            start_index = np.random.randint(0, music_len - self.sample_len + 1)
            music = music[start_index:start_index+self.sample_len, :]
            h = h[start_index:start_index+self.sample_len]
        elif music_len < self.sample_len:
            music = np.concatenate([music, np.zeros([self.sample_len-music_len,music.shape[1]])+pad], axis=0)
            h_pad = np.min(h) # use minimum value as pad so that it would not influence the norm operation
            h = np.concatenate([h, np.zeros([self.sample_len-music_len])+h_pad], axis=0)

        music = music[::(self.gap+1)]
        h = h[::(self.gap+1)]
        return music, h

def load_data(root_path, train_prop = 0.9, max_len = 600, gap = 0):
    file_path = []
    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file[-4:] == ".pkl":
                file_path.append(os.path.join(dirpath, file))
    if train_prop == 1.0:
        return ML_Dataset(file_path, max_len, gap)
    else:
        train_num = round(len(file_path) * train_prop)
        # np.random.shuffle(file_path)
        return ML_Dataset(file_path[:train_num], max_len, gap), ML_Dataset(file_path[train_num:], max_len, gap)

if __name__ == '__main__':
    # test
    train_data, test_data = load_data("./test/data",train_prop=0.5,pretrain=False)
    print(len(train_data))
    print(len(test_data))
    # print(train_data[0].shape)
    print(train_data[0][0].shape)
    print(train_data[0][1].shape)