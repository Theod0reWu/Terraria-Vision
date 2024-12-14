"""Creates the tiles in test_build_1 for classification"""

from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import torch
import os
import numpy as np

from sprite_tools import get_sprite_custom, get_sprite

TILE_POS = [(e, i) for e in [0, 1, 5] for i in range(6)] + [(e, i) for e in range(2, 5) for i in [0, 1, 5]]

class TestDataset(Dataset):

	idx_map = {i : TILE_POS[i] for i in range(len(TILE_POS))}

	def __init__(self, path = "../images/test_builds/test_build_1.png"):
		'''
		    Test dataset made from test_buld_1.png image
		'''
		self.path = path
		self.tensor_image = torch.moveaxis(read_image(path)[:, 34:, 30:], 0, -1)

		self.section_rows = 12
		self.section_cols = 12

		self.rows = 6
		self.cols = 6

		self.length = 12 * 12 - 7
		self.section_length = 25

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
		return get_sprite(section, row, col)

	def __len__(self):
	    return self.length

	def id_to_img(self, idx):
		pass

	def __getitem__(self, idx):
		section_idx = idx // 30
		tile_idx = idx % 30

		tile_row, tile_col = TestDataset.idx_map[tile_idx]

		if section_idx >= 9 * 12 + 4:
			section_idx += 1
		section_row = section_idx // 12
		section_col = section_idx % 12

		section = self.get_section(section_row, section_col)

		return self.get_tile(tile_row, tile_col, section)


