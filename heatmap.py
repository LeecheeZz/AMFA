import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import os
# import sys
# sys.path.insert(0,"./")
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
# from tool.utils import load_network
import yaml
import argparse
import torch
from torchvision import datasets, models, transforms
from PIL import Image
parser = argparse.ArgumentParser(description='Training')
import math
# from models.ConvNext import make_convnext_model
from timm import create_model
# from timm.models import create_model
from sample4geo.hand_convnext.model import two_view_net

parser.add_argument('--data_dir',default='/home/imt3090/huangshuheng/dataset/U1652/shift_satellite',type=str, help='./test_data')
parser.add_argument('--batchsize', default=1, type=int, help='batchsize')
parser.add_argument('--checkpoint',default="model/DML", help='weights' )
# parser.add_argument('--platform',default="satellite", help='weights' )
parser.add_argument('--platform',default="drone", help='weights' )
opt = parser.parse_args()

config_path = 'checkpoints/university/opts.yaml'
with open(config_path, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
for cfg, value in config.items():
    if cfg not in opt:
        setattr(opt, cfg, value)


def heatmap2d(img, arr):
    # fig = plt.figure()
    # ax0 = fig.add_subplot(121, title="Image")
    # ax1 = fig.add_subplot(122, title="Heatmap")
    # fig, ax = plt.subplots(）
    # ax[0].imshow(Image.open(img))
    plt.figure()
    heatmap = plt.imshow(arr, cmap='viridis')
    plt.axis('off')
    # fig.colorbar(heatmap, fraction=0.046, pad=0.04)
    #plt.show()
    plt.savefig('heatmap_dbase')

data_transforms = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

from sample4geo.hand_convnext.model import make_model
model = make_model(opt)
model = model.model_1


# 权重路径
load_from = "checkpoints/sues-200/convnext_base.fb_in22k_ft_in1k_384/0221_11:36:17/cbam.pth"
# model.load_state_dict(torch.load(load_from))

pretran_model = torch.load(load_from, map_location='cuda:0')
model2_dict = model.state_dict()
state_dict = {k: v for k, v in pretran_model.items() if k in model2_dict.keys() and v.size() == model2_dict[k].size()}
model2_dict.update(state_dict)
model.load_state_dict(model2_dict)

model = model.eval().cuda()
#---------------------------图片文件夹路径-----------------------------
name = "drone"
folder_path = "/home/imt3090/huangshuheng/dataset/U1652"
#---------------------------------------------------------------------
folder_path = os.path.join(folder_path, name)

image_names = [
        name for name in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, name))
    ]
image_names = sorted(image_names)
# print("子文件夹名称列表：", subfolder_names)
# print(len(subfolder_names))

print(folder_path)
for i in image_names: 
    print(i)
    # imgpath = os.path.join(opt.data_dir,"gallery_{}/{}".format(opt.platform,i))
    imgpath = os.path.join(folder_path,  i) 
    print(imgpath)
    img = Image.open(imgpath)
    img = data_transforms(img)
    img = torch.unsqueeze(img,0)

    with torch.no_grad():
        # print(model)
        features = model.convnext(img.cuda())
        # features = features[1]
        features = model.amfa(features[0])
        # tri_features = model.se_attention2(features[1]) #[8, 768, 8, 8]  [8, 768, 8, 8]
        # gap_feature = model.se_attention1(features[0]) # [8, 768, 8, 8]
        # trise_features = (gap_feature + tri_features[0] + tri_features[1]) / 3.0

        # rep_features = model.rep(trise_features)

        # final_feature = (features[0] + rep_features) / 2.0
        # final_feature = trise_features

        # pos_embed = model.backbone.pos_embed
        # if opt.backbone=="resnet50":
        #     output = features
        # else:
        # part_features = features[:,1:]
        # part_features = features[0]
        # part_features = part_features.view(part_features.size(0),int(math.sqrt(part_features.size(1))),int(math.sqrt(part_features.size(1))),part_features.size(2))
        # output = part_features.permute(0,3,1,2) # B C H W
        output = features

    heatmap = output.squeeze().sum(dim=0).cpu().numpy()
    # print(heatmap.shape)
    # print(heatmap)
    # heatmap = np.mean(heatmap, axis=0)
    #
    # heatmap = np.maximum(heatmap, 0)
    heatmap = normalization(heatmap)
    img = cv2.imread(imgpath)  # 用cv2加载原始图像 
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, 2)  # 将热力图应用于原始图像model.py
    ratio = 0.50 if opt.platform == "drone" else 0.40
    superimposed_img = heatmap * ratio + img  # 这里的0.4是热力图强度因子
    if not os.path.exists("CBAM_" + name):
        os.mkdir("CBAM_" + name)
    save_file = "CBAM_" + name
    save_file = save_file + "/{}.jpg".format(os.path.splitext(i)[0])
    cv2.imwrite(save_file, superimposed_img)