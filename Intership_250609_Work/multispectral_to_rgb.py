import os
import rasterio
import numpy as np
from PIL import Image

# 输入和输出文件路径
input_file = r'E:\QingHua_IntershipRepository_res\res\pic\2019_1101_nofire_B2348_B12_10m_roi.tif'
output_dir = r'E:\QingHua_IntershipRepository\Result\0609_RGB_Conversion'
output_file = os.path.join(output_dir, 'RGB_image.png')

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 打开输入TIFF文件
with rasterio.open(input_file) as src:
    # 假设文件包含B2、B3、B4波段, 我们选择B2、B3、B4作为RGB
    band_blue = src.read(1)    # 波段2
    band_green = src.read(2)   # 波段3
    band_red = src.read(3)     # 波段4

# 根据原范围[0, 10000]缩放到[0, 255]
def rescale_bands(band):
    return np.clip((band - band.min()) * (255.0 / (10000.0 - band.min())), 0, 255).astype(np.uint8)

# 缩放RGB通道
red_scaled = rescale_bands(band_red)
green_scaled = rescale_bands(band_green)
blue_scaled = rescale_bands(band_blue)

# 创建RGB数组
rgb = np.stack((red_scaled, green_scaled, blue_scaled), axis=2)

# 用PIL保存为PNG文件
rgb_image = Image.fromarray(rgb, 'RGB')
rgb_image.save(output_file)

# 可选：显示图像验证效果
rgb_image.show()

print(f"RGB图像已保存到: {output_file}")
