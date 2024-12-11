from torch.utils.data import Dataset
import os
import pickle
import numpy as np

pad = -1000

class ML_Dataset(Dataset):
    def __init__(self, file_path, max_len = 600, gap = 0, fix_start = None):
        self.file_path = file_path
        self.max_len = int(max_len)
        self.gap = int(gap)
        self.sample_len = self.max_len * (gap + 1)
        self.fix_start = fix_start

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

        h = [light[int(i)][60][0] for i in sample]
        v = [light[int(i)][0][1] for i in sample]
        h = np.array(h)
        # print('h shape', h.shape)
        v = np.array(v)
        # print('v shape', v.shape)

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
            # h_pad = np.min(h) # use minimum value as pad so that it would not influence the norm operation
            h = np.concatenate([h, np.zeros([self.sample_len-music_len])+pad], axis=0)
            # v_pad = np.min(v)
            v = np.concatenate([v, np.zeros([self.sample_len-music_len])+pad], axis=0)

        music = music[::(self.gap+1)]
        h = h[::(self.gap+1)]
        v = v[::(self.gap+1)]
        hv = np.stack([h, v], axis=1)

        f_name = self.file_path[index].split('/')[-1]
        return music, hv, f_name

def load_data(root_path, train_prop = 0.9, max_len = 600, gap = 0, shuffle = False, random_seed = 42, fix_start = None):
    file_path = []
    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file[-4:] == ".pkl":
                file_path.append(os.path.join(dirpath, file))
    if train_prop == 1.0:
        return ML_Dataset(file_path, max_len, gap, fix_start)
    else:
        train_num = round(len(file_path) * train_prop)
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(file_path)
        return ML_Dataset(file_path[:train_num], max_len, gap, fix_start), ML_Dataset(file_path[train_num:], max_len, gap, fix_start)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # test
    train_data, test_data = load_data("./test/data/",train_prop=0.5, fix_start=40)
    # print(len(train_data))
    # print(len(test_data))
    # print(train_data[0][0].shape)
    # print(train_data[0][1].shape)
    print(train_data[0][2])
        
    