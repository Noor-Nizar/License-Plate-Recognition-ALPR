from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from glob import glob
import numpy

from .logger_config import logger

class CustomDataset(Dataset):
    ''' Custom dataset class for loading images and annotations '''
    def __init__(self, image_paths, bboxes):
        self.image_paths = image_paths
        self.bboxes = bboxes
        self.pad_to_640 = lambda x: x.resize((640, 640))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        crops = []
        for bbox in self.bboxes[idx]:
            crop = img.crop(bbox.detach().cpu().numpy())
            crop = self.pad_to_640(crop)
            crops.append(crop)
        
        return np.array(crops)
        # img = numpy.array(img)
        # #pad to 640x640
        # h, w, _ = img.shape
        # # logger.info(f"Image shape: {img.shape}")
        # if h < 640:
        #     pad = np.zeros((640-h, w, 3))
        #     img = np.concatenate((img, pad), axis=0)
        # if w < 640:
        #     pad = np.zeros((640, 640-w, 3))
        #     img = np.concatenate((img, pad), axis=1)

        # # logger.info(f"Image shape: {img.shape}")
        # return img

def collate_fn(batch):
    ## concatenate on the first axis
    return np.concatenate(batch, axis=0)

def get_dataset_loader(img_paths, bboxes, batch_size=8):
    dataset = CustomDataset(img_paths, bboxes, collate_fn=collate_fn)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataset, dataloader

