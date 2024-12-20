from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision import transforms
import torch
import os
import random

IGNORE = {5, 15, 81, 95, 126, 133, 171, 216, 323, 324, 406, 442, 552, 567, 583, 584, 585, 586, 587, 588, 589, 596, 616, 634}

class ImageData():

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
        self.dirnums = []

        num_tiles = len(os.listdir(path))
        current = 0
        for tile_num in range(num_tiles):
            dirname = "Tiles_" + str(tile_num)
            path_to_dir = os.path.join(path, dirname)

            if tile_num not in IGNORE and os.path.exists(path_to_dir):
                dir_len = len(os.listdir(path_to_dir))
                if (dir_len > 0):
                    self.ranges.append((current, current + dir_len - 1))
                    current += dir_len
                    self.dirnames.append(dirname)
                    self.dirnums.append(tile_num)
        self.length = current

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

    def load_image(self, tile_idx, img_idx):
        path_name = os.path.join(self.path, self.dirnames[tile_idx], f"{img_idx}.png")
        tensor_image = read_image(path_name)

        # normalize pixel values
        tensor_image = tensor_image.float() / 255.0
        if (tensor_image.shape != torch.Size([3, 8, 8])):
            print(self.dirnames[tile_idx], img_idx)
        return tensor_image

class TileDataset(ImageData):
    '''
        Not a torch dataset
    '''
    def __init__(self, path):
        super(TileDataset, self).__init__(path)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.idx_to_img(idx)
        return self.load_image(img[0], img[1]), self.dirnums[img[0]]


class PairDataset(ImageData, Dataset):
    def __init__(self, path):
        super(PairDataset, self).__init__(path)

    def __len__(self):
        return self.length * self.length - 1

    def __getitem__(self, idx):
        first, second = idx % self.length, idx // self.length
        first_img, second_img = self.idx_to_img(first), self.idx_to_img(second)
        similarity = torch.tensor([first_img[0] == second_img[0]])
        return self.load_image(first_img[0], first_img[1]), self.load_image(second_img[0], second_img[1]), similarity

if __name__ == "__main__":
    pd = PairDataset("../dataset/")
    # for i in range(pd.length + 1):
    #     pd[i]

    d = TileDataset("../dataset/")
    print(len(d))