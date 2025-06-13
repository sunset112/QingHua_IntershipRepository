import os
from torch.utils.data import Dataset
from PIL import Image

class ImageTxtDataset(Dataset):
    def __init__(self, txt_path: str, folder_name, transform):
        self.transform = transform
        self.data_dir = os.path.dirname(txt_path)
        self.imgs_path = []
        self.labels = []
        self.folder_name = folder_name
        
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"找不到文件：{txt_path}")
            
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            try:
                img_path, label = line.strip().split()
                label = int(label)
                img_path = os.path.join(self.data_dir, self.folder_name, img_path)
                
                if not os.path.exists(img_path):
                    print(f"警告：图像文件不存在：{img_path}")
                    continue
                    
                self.labels.append(label)
                self.imgs_path.append(img_path)
            except ValueError as e:
                print(f"警告：无法解析行：{line.strip()}, 错误：{str(e)}")
            except Exception as e:
                print(f"处理行时出错：{line.strip()}, 错误：{str(e)}")

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, i):
        path, label = self.imgs_path[i], self.labels[i]
        try:
            image = Image.open(path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"加载图像时出错：{path}, 错误：{str(e)}")
            # 返回一个空图像和标签
            image = Image.new('RGB', (256, 256))
            if self.transform is not None:
                image = self.transform(image)
            return image, label
