from torch.utils.data import DataLoader, Dataset
import torch
import os
import numpy as np
from torchvision.io import read_image
from torchvision import transforms

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
        self.indices = []
        self.transforms = transforms.Resize((8, 8))

        num_tiles = len(os.listdir(path))
        current = 0
        for tile_num in range(num_tiles):
            dirname = "Tiles_" + str(tile_num)
            pathname = os.path.join(path, dirname)
            if os.path.exists(pathname):
                dir_len = len(os.listdir(pathname))
                if (dir_len > 0):
                    self.ranges.append([current, current + dir_len - 1])
                    current += dir_len
                    self.dirnames.append(dirname)
        self.length = current
        self.ranges = np.array(self.ranges)

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
    
    def load_img(self, dir_idx, offset):
        path_name = os.path.join("..", "dataset", self.dirnames[dir_idx], f"{offset}.png")
        tensor_image = read_image(path_name)

        # normalize pixel values
        tensor_image = tensor_image.float() / 255
        # resize to 8x8 (while training found few images that were 6x8)
        tensor_image = self.transforms(tensor_image)
        return tensor_image

    """
    def __getitem__(self, idx):
        first, second = idx % self.length, idx // self.length
        first_img, second_img = self.idx_to_img(first), self.idx_to_img(second)
        similarity = torch.tensor([first_img[0] == second_img[0]])
        
        return first_img, second_img, similarity
    """
    def __getitem__(self, _):
        # we randomly choose to either get a similar or dissimilar image
        if np.random.uniform(0, 1) < 0.5:
            # the chance of picking the same folder is fairly small so we can 
            # assume that this will give us different tile index
            tr_idxs = np.random.randint(0, len(self.ranges), size=2)
        else:
            # else we get the two images from the same folder
            tr_idxs = np.random.randint(0, len(self.ranges), size = (1))[[0,0]]

        tiles = self.ranges[tr_idxs] - self.ranges[tr_idxs][:,0].reshape(-1, 1)
        # randomly choose image indices in their respecitve folders
        images = []
        for i in range(2):
            offset = 0
            if tiles[i][1]:
                offset = np.random.randint(*tiles[i])
            # read in images from disk and load into tensors
            img = self.load_img(tr_idxs[i], offset)
            images.append(img)
        
        # our label, if they're in the same folder label is 1 else 0
        similarity = torch.tensor(tr_idxs[0] == tr_idxs[1], dtype=torch.float32)

        return images[0], images[1], similarity

if __name__ == "__main__":
    pd = PairDataset(os.path.join("..", "dataset"))
    print(pd.ranges)
    print(pd.idx_to_img(0))
    print(pd.idx_to_img(pd.length - 1))
    print(pd.idx_to_img(100))
    print(pd.idx_to_img(200))
    print(pd.idx_to_img(500))