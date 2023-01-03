import os
from tensorflow.keras.preprocessing import image
import numpy as np
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("No such module has module named 'train_test_split'")


class DataLoader():
    def __init__(self,data_path, width, height):
        self.data_path = data_path
        # self.data_label = data_label
        self.height = height
        self.width = width
        self.__labels = []

    def __len__(self):
        return len(self.data_path)

    def load_data(self):
        labels = os.listdir(self.data_path)
        images = []

        for label in labels:
            self.__labels.append(label)
            curr = os.path.join(self.data_path, label)

            for img in os.listdir(curr):
                full_path = os.path.join(curr, img)
                img = image.load_img(full_path ,target_size=(self.width,self.height), color_mode='grayscale')
                img = image.img_to_array(img)
                img = img / 255.0
                images.append(img)

        images = np.array(images)

        return images
    
    def get_lables(self):
        return self.__labels

    def data_gen(self,image_list, size):
        return train_test_split(image_list , test_size = size)