import cv2
import os
import pickle
import torch
import numpy as np
from siamese_model import EmbeddingNetwork
from PIL import Image

class TileClassifier:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.clf = pickle.load(f)
        class_names = []
        with open("TileID.txt", "r") as f:
            for line in f:
                class_names.append(line.split("\t")[0])

        #self.clf.classes_ = np.array(class_names)
        model = EmbeddingNetwork()
        model_path_name = os.path.join("..", "models", "embedding_network.pth")
        weights = torch.load(model_path_name)
        model.load_state_dict(weights)
        self.embedding_model = model

    
    def __call__(self, img, k):
        img = torch.tensor(img)
        img = img.type(torch.float32)
        img = img.permute((2, 0, 1))
        img = img.view((1, 3, 8, 8))
        embedding = self.embedding_model(img).detach().numpy()
        probs = self.clf.predict_proba(embedding)[0]
        probs, classes = list(zip(*sorted(zip(probs, self.clf.classes_), reverse=True)))
        return classes[:k], probs[:k]

class BaseClassifier:
    def __init__(self, tile_classifier, classification_threshold, top_k):
        self.classification_threshold = classification_threshold
        self.tile_classifier = tile_classifier
        self.top_k = top_k

    def classify_tile(self, img):
        img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_LINEAR)
        top_k_labels, probs = self.tile_classifier(img, self.top_k)
        return top_k_labels, probs
    
    def segment_base(self, base_image):
        window_size = base_image.shape[0] // 10
        frac_matches = 0
        opt_window_size = window_size
        best_frac_matches = 0
        while(window_size > 1):

            frac_matches = 0
            num_tiles = 0
            for i in range(0, base_image.shape[0] + 1, window_size):
                for j in range(0, base_image.shape[1] + 1, window_size):
                    window_length = min(base_image.shape[0],i+window_size)
                    window_width = min(base_image.shape[1],j+window_size)
                    img = base_image[i:window_length, j:window_width]
                    if img.shape[0] >= 8 and img.shape[1] >= 8:
                        _, probs = self.classify_tile(img)
                        if probs[0] > self.classification_threshold:
                            frac_matches += 1
                    num_tiles += 1
            
            frac_matches /= num_tiles
            if frac_matches < best_frac_matches:
                best_frac_matches = frac_matches
                opt_window_size = window_size
            window_size -= 1
        
        return opt_window_size
    
    def classify_base(self, base_image):
        window_size = self.segment_base(base_image)
        classification_results = []
        tile_index = 0
        for i in range(0, base_image.shape[0] + 1, window_size):
            for j in range(0, base_image.shape[1] + 1, window_size):
                window_length = min(base_image.shape[0],i+window_size)
                window_width = min(base_image.shape[1],j+window_size)

                center_x = i + (window_length - i) // 2
                center_y = j + (window_width - j) // 2

                img_tile = base_image[i:window_length, j:window_width]
                top_k, probs = self.classify_tile(img_tile)

                classification_results.append({
                    'index': tile_index,
                    'center': (center_x, center_y),
                    'top_k': top_k,
                    'probs': probs
                })
                tile_index += 1

        return window_size, classification_results
    
    def log_classifications(self, base_image):
        window_size, classification_results = self.classify_base(base_image)

        annotated_image = base_image.copy()
        annotated_img_path = os.path.join("..", "test_logs", "annotated_image.png")
        classification_results_path = os.path.join("..", "test_logs", "classification_results.txt")

        tile_index = 0
        for i in range(0, base_image.shape[0], window_size):
            for j in range(0, base_image.shape[1], window_size):
                cv2.rectangle(annotated_image, 
                              (j, i), 
                              (min(base_image.shape[1], j + window_size), 
                               min(base_image.shape[0], i + window_size)), 
                              color=(0, 0, 255),
                              thickness=1)

                center_x = i + (min(base_image.shape[0], i + window_size) - i) // 2
                center_y = j + (min(base_image.shape[1], j + window_size) - j) // 2
                cv2.putText(annotated_image, str(tile_index), 
                            (center_y, center_x), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=0.5, 
                            color=(255, 255, 255),
                            thickness=1)
                tile_index += 1

        cv2.imwrite(annotated_img_path, annotated_image)

        with open(classification_results_path, 'w') as f:
            for result in classification_results:
                f.write(f"Tile Index: {result['index']}\n")
                f.write(f"Center: {result['center']}\n")
                f.write(f"Top-K Classes: {result['top_k']}\n")
                f.write(f"Probabilities: {result['probs']}\n")
                f.write("\n")
    
    def __call__(self, base_img):
        self.log_classifications(base_img)


if __name__ == "__main__":
    model_path = os.path.join("..", "models", "model.pkl")
    clf = TileClassifier(model_path)
    base_classifier = BaseClassifier(clf, .4, 2)

    img_path = os.path.join("..", "images", "test_builds", "test_build_1.png")
    base_image = cv2.imread(img_path)
    print(base_image.shape)
    base_classifier.log_classifications(base_image)