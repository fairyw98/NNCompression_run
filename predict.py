import csv
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pickle
from models.alexnet.predict_test import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "img_23670.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)


    partition_id = [3,6,10]
    quantization = [1,2,4,8,16,32]
    coder_channels = [1,2,4,8,16,32,64]
    en_stide = [1,2,3,5,6,7,9]
    
    # partition_id = [10]
    # quantization = [32]
    # coder_channels = [64]
    # en_stide = [2]

    # with open('database5.csv','w',newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['partition_id','quantization','coder_channels','en_stride','size'])
    # for p in partition_id:
    #     for q in quantization:
    #         for c in coder_channels:
    #             for e in en_stide:
    #                 model = AlexNet(partition_id = p, quant_bits = q, coder_cfg = {'coder_channels':c,'en_stride':e}).to(device)
    #                 model.eval()
    #                 with torch.no_grad():
    #                     # predict class
    #                     output = model(img.to(device))
    #                     ofile = open("BinaryData",'wb')
    #                     pickle.dump(output, ofile)
    #                     ofile.close()
    #                     size = os.path.getsize("BinaryData")
    #                     print(size)

    #                 with open('database5.csv','a+',newline='') as f:
    #                     writer = csv.writer(f)
    #                     writer.writerow([p,q,c,e,size])
    # create model
    model = AlexNet(num_classes=5).to(device)
    from torchvision.utils import save_image
    model.eval()
    with torch.no_grad():
        # predict class
        # output = model(img.to(device))
        # e = torch.stack((output),dim=0)
        # print(type(output))
        # print(output.shape)
        output = torch.randn(1,32,11,11)
        save_image(output,'test.png')
        # ofile = open("BinaryData",'wb')
        # pickle.dump(output, ofile)
        # # str=pickle.dumps(output)
        # ofile.close()
        # print(os.path.getsize("BinaryData")) 
        # # print(type(str))
        # print(output.shape)
    # from torchvision.utils import save_image

    # ...
    # save_image(im, f'im_name.png')
# import os, torch  
# import matplotlib.pyplot as plt 
# from time import time 
# from torchvision import utils 

# read_path = 'D:test'
# write_path = 'D:test\\write\\'
 
# # matplotlib 读取 2
# start_t = time()
# imgs = torch.zeros([5, 1080, 1920, 3], device='cuda')
# for img, i in zip(os.listdir(read_path), range(5)): 
#   img = torch.tensor(plt.imread(os.path.join(read_path, img)), device='cuda')
#   imgs[i] = img    
# imgs = imgs.permute([0,3,1,2])/255  
# print('matplotlib 读取时间2：', time() - start_t) 
# # torchvision 保存
# start_t = time() 
# for i in range(imgs.shape[0]):   
#   utils.save_image(imgs[i], write_path + str(i) + '.jpg')
# print('torchvision 保存时间：', time() - start_t)

if __name__ == '__main__':
    # main()
    import torch
    import torchvision
    import numpy as np
    from PIL import Image

    # 读取rgb格式的图片
    img = torch.randn(1,1,11*32,11)
    # img = torch.Tensor(img)
    print(img.shape)
    # ofile = open("BinaryData",'wb')
    # pickle.dump(img, ofile)
    # print(os.path.getsize("BinaryData")) 
    # 以下两句代码可以注释，save_image()函数里已经包含了make_grid()操作
    # img_grid = torchvision.utils.make_grid(img)
    # print(img_grid.shape)

    # img如果没有归一化，必须要除以255。
    torchvision.utils.save_image(img,"test.png")
    print(os.path.getsize("test.png")) 
    ofile = open("BinaryData",'wb')
    pickle.dump(img, ofile)
    ofile.close()
    print(os.path.getsize("BinaryData")) 