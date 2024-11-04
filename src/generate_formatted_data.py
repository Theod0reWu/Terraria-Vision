import os
import re

import cv2

from sprite_tools import display, get_sprite, downsample, is_blank, get_dimensions

def get_images(directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']  
    image_files = []
    for filename in os.listdir(directory):
        if any(filename.endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(directory, filename))
    return image_files

directory_path = '../images/terraria_tiles/'
image_list = get_images(directory_path)

save_path = '../dataset/'
if not os.path.exists(save_path):
	os.makedirs(save_path)

for image_path in image_list:
	img = cv2.imread(image_path)
	rows, cols = get_dimensions(img)

	# create_directory for each sprite sheet
	filename = os.path.splitext(os.path.basename(image_path))[0]
	filename = re.sub(r'[^0-9]+$', '', filename)
	new_dir = os.path.join(save_path, filename)
	if not os.path.exists(new_dir):
		os.makedirs(new_dir)
	
	num = 0
	for r in range(int(rows)):
		for c in range(int(cols)):
			sprite = get_sprite(img, r, c)

			if not is_blank(sprite):
				display(sprite, os.path.join(new_dir, str(num)+".png"))
				num += 1


