from torchvision import transforms as T
from base import BaseDataLoader
from .datasets import RefCOCODataset

def get_transform(img_size):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class RefCOCODataLoader(BaseDataLoader):

    def __init__(self, label_path, img_folder, config, batch_size=None, 
                 shuffle=True, validation_split=0.0, num_workers=0):
        
        transform = get_transform(config.img_size)
        
        # Khởi tạo dataset với các tham số mới
        self.dataset = RefCOCODataset(label_path, img_folder, config, transform)
        
        super().__init__(
            self.dataset,
            batch_size=batch_size or config.batch_size,
            shuffle=shuffle,
            validation_split=validation_split,
            num_workers=num_workers
        )