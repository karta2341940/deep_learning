import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, image_dir, mask_dir, batch_size, image_size, num_classes):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.image_filenames = os.listdir(self.image_dir)
        self.mask_filenames = os.listdir(self.mask_dir)
        
    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / self.batch_size))
    
    def __getitem__(self, index):
        batch_image_filenames = self.image_filenames[index * self.batch_size: (index + 1) * self.batch_size]
        batch_mask_filenames = self.mask_filenames[index * self.batch_size: (index + 1) * self.batch_size]
        
        batch_images = []
        batch_masks = []
        
        for image_filename, mask_filename in zip(batch_image_filenames, batch_mask_filenames):
            image_path = os.path.join(self.image_dir, image_filename)
            mask_path = os.path.join(self.mask_dir, mask_filename)
            
            image = load_img(image_path, target_size=self.image_size)
            mask = load_img(mask_path, target_size=self.image_size, color_mode='grayscale')
            
            image = img_to_array(image) / 255.0
            mask = img_to_array(mask) / 255.0 
            batch_images.append(image)
            batch_masks.append(mask)
        
        batch_images = np.array(batch_images)
        batch_masks = np.array(batch_masks)
        #batch_masks = to_categorical(batch_masks, num_classes=self.num_classes)
        
        return batch_images, batch_masks
# 測試程式碼
image_dir = './data/imgs/'  # 圖像資料夾的路徑
mask_dir = './data/masks/'  # 遮罩資料夾的路徑
batch_size = 4  # 批次大小
image_size = (256, 256)  # 圖像大小
num_classes = 2  # 分類的類別數量

data_generator = DataGenerator(image_dir, mask_dir, batch_size, image_size, num_classes)
batch_images, batch_masks = data_generator[0]

print('Batch Images Shape:', batch_images.shape)
print('Batch Masks Shape:', batch_masks.shape)
