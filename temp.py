import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 读取图片
image_path = 'D:/code/DWPose-onnx/output2/frame_0001.jpg'  # 替换为您的图片路径
image = Image.open(image_path)

# 定义转换操作
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整大小
    transforms.ToTensor(),           # 转换为张量，并将范围从 [0, 255] 转换为 [0.0, 1.0]
])

# 应用转换
image_tensor = transform(image)

# 将张量的形状调整为 (3, 224, 224)
# image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度，如果需要

# 打印张量形状
print(f'Image Tensor Shape: {image_tensor.shape}')  # 应该是 (1, 3, 224, 224)
image_np = image_tensor.permute(1, 2, 0).numpy()
# 显示图片
plt.imshow(image_np)  # 将形状调整为 (H, W, C) 以便显示
plt.axis('off')  # 不显示坐标轴
plt.show()