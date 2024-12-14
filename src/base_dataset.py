"""Dataset made from test_build_1.png"""

from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import torch
import os
import numpy as np
import csv

from sprite_tools import get_sprite_custom, get_sprite

TILE_POS = [(e, i) for e in [0, 1, 5] for i in range(6)] + [(e, i) for e in range(2, 5) for i in [0, 1, 5]]
IGNORE_SECTIONS = [(9, 4), (3, 2), (3, 3), (3, 4), (3, 5), (4, 0), (4, 1)]

class TestDataset(Dataset):

	idx_map = {i : TILE_POS[i] for i in range(len(TILE_POS))}

	def __init__(self, path = "../images/test_builds/test_build_1.png", label_path = "../images/test_builds/test_build_1_labels.csv"):
		'''
		    Test dataset made from test_buld_1.png image
		'''
		self.path = path
		self.tensor_image = torch.moveaxis(read_image(path)[:, 34:, 30:], 0, -1)

		# Replace 'your_file.csv' with your actual file name
		self.label_path = label_path
		with open(label_path, 'r') as file:
			csv_reader = csv.reader(file)
			self.labels = [row for row in csv_reader]

		self.section_rows = 11
		self.section_cols = 13

		self.rows = 6
		self.cols = 6

		self.section_length = 27
		self.length = (self.section_cols * self.section_rows - 6 - len(IGNORE_SECTIONS)) * self.section_length

	def get_section(self, row, col):
	    """
	    Gets the sections with an offset of 16 horizontally and 32 vertically. Each section should be a 6*16 square 
	    """
	    assert row < self.section_rows and col < self.section_cols
	    return get_sprite_custom(self.tensor_image, row, col, 6 * 16, 32, 16)

	def get_tile(self, row, col, section):
	    """
	    Expected section to contain blocks in the format:

	        ######
	        ######
	        ##   #
	        ##   #
	        ##   #
	        ######

	    Where # is a block

	    """
	    assert row < self.rows and col < self.cols
	    return get_sprite(section, row, col, offset = 0)

	def __len__(self):
	    return self.length

	def id_to_img(self, idx):
	    pass


	def __getitem__(self, idx):
		section_idx = idx // self.section_length
		tile_idx = idx % self.section_length

		tile_row, tile_col = TestDataset.idx_map[tile_idx]

		for i in IGNORE_SECTIONS:
			if section_idx >= i[0] * self.section_cols + i[1]:
				section_idx += 1
		    
		section_row = section_idx // self.section_cols
		section_col = section_idx % self.section_cols

		section = self.get_section(section_row, section_col)

		return self.get_tile(tile_row, tile_col, section), self.labels[section_row][section_col]