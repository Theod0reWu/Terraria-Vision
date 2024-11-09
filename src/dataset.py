from torch.utils.data import DataLoader, Dataset
import torch
import os
import random
from torchvision.io import read_image

class PairDataset(Dataset):
    def __init__(self, path):
        '''
            Creates the pair dataset for sprites. 

            Expects "path" to be a directory containing numbered directories of the form:
                Tile_0
                ...
                Tile_100
        '''
        self.path = path
        self.length = 0
        self.ranges = []
        self.dirnames = []

        num_tiles = len(os.listdir(path))
        current = 0
        for tile_num in range(num_tiles):
            dirname = "Tiles_" + str(tile_num)
            dir_len = len(os.listdir(os.path.join(path, dirname)))
            if (dir_len > 0):
                self.ranges.append((current, current + dir_len - 1))
                current += dir_len
                self.dirnames.append(dirname)
        self.length = current

    def __len__(self):
        return self.length

    
    def idx_to_img(self, idx):
        low, high = 0, len(self.ranges)
        while (low < high):
            mid = (low + high) // 2
            mid_range = self.ranges[mid]

            if (mid_range[0] <= idx and idx <= mid_range[1]):
                return mid, idx - mid_range[0]
            elif (idx > mid_range[1]):
                low = mid + 1
            else:
                high = mid
        return None, None
    
    def load_img(self, range, offset):
        path_name = os.path.join("..", "dataset", f"Tiles_{range}", f"{offset}.png")
        tensor_image = read_image(path_name)

        # normalize pixel values
        tensor_image = tensor_image.float() / 255
        return tensor_image

    """
    def __getitem__(self, idx):
        first, second = idx % self.length, idx // self.length
        first_img, second_img = self.idx_to_img(first), self.idx_to_img(second)
        similarity = torch.tensor([first_img[0] == second_img[0]])
        
        return first_img, second_img, similarity
    """
    def __getitem__(self, idx):
        if random.uniform(0, 1) < 0.5:
            first, second = idx % self.length, idx // self.length
            first_img, second_img = self.idx_to_img(first), self.idx_to_img(second)
        else:
            tr_idx = random.randint(0, len(self.ranges))
            offsets = random.choices(range(*self.ranges[tr_idx]), k=2)
            first_img, second_img = zip([tr_idx]*2, offsets)
        
        similarity = torch.tensor(first_img[0] == second_img[0])
        first_img = self.load_img(first_img[0], first_img[1])
        second_img = self.load_img(second_img[0], second_img[1])
        return first_img, second_img, similarity

pd = PairDataset(os.path.join("..", "dataset"))
print(pd.ranges)
print(pd.idx_to_img(0))
print(pd.idx_to_img(pd.length - 1))
print(pd.idx_to_img(100))
print(pd.idx_to_img(200))
print(pd.idx_to_img(500))