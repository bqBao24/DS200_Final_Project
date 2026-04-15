import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import BertTokenizer

class RefCOCODataset(Dataset):    
    def __init__(self, label_path, img_folder, config, transform=None):
        self.data = torch.load(label_path)
        self.img_folder = img_folder
        self.config = config
        self.transform = transform
        
        self.tokenizer = BertTokenizer.from_pretrained(config.text_encoder)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
            item = self.data[idx]
            img_name = item[0]   
            bbox_raw = item[2]   
            text = item[3]      

            img_path = os.path.join(self.img_folder, img_name)
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size

            x1, y1, x2, y2 = bbox_raw
            bbox = torch.tensor([x1/orig_w, y1/orig_h, x2/orig_w, y2/orig_h], dtype=torch.float32)

            if self.transform:
                img = self.transform(img)

            tokens = self.tokenizer(text, max_length=self.config.max_text_len,
                                    padding='max_length', truncation=True, return_tensors='pt')

            return {
                'image': img,
                'input_ids': tokens['input_ids'].squeeze(0),
                'attention_mask': tokens['attention_mask'].squeeze(0),
                'bbox': bbox,
                'text': text
            }