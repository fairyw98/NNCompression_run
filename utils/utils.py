import os
import sys
import json
import pickle
import random
import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

def Quant(detach_x,quant_bits):
    _x = detach_x
    for _ in range(_x.shape[1]):
        _max=torch.max(_x[:,_])
        _min=torch.min(_x[:,_])         
        code_book=torch.round((_x[:,_]-_min)*quant_bits/(_max-_min))
        _x[:,_]=_min+code_book*((_max-_min)/quant_bits)
    return _x

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
    #  torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.deterministic = True

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    # print("{} images were found in the dataset.".format(sum(every_class_num)))
    # print("{} images for training.".format(len(train_images_path)))
    # print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def Cluster(train_tmp = [],cluster_num = 3):
    train_tmp2 = train_tmp.copy()
    random.Random(6).shuffle(train_tmp2)
    chunk_size = int(len(train_tmp2)/cluster_num)+1
    FLAGS_cluster_set = [train_tmp2[i:i + chunk_size] for i in range(0, len(train_tmp2), chunk_size)]
    return FLAGS_cluster_set


def sw_train_one_epoch(model, optimizer, data_loader, device, epoch,train_tmp = []):
    # print(train_tmp)
    model.train()
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout,leave=True)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]

        # sel_train_tmp = [random.choice(set_tmp) if isinstance(set_tmp[0],list) else set_tmp for set_tmp in train_tmp]
        sel_train_tmp = [random.choice(train_tmp) for i in range(3)]
        len_train_tmp = len(sel_train_tmp)
        for index,(_coder_channels, _en_stride) in enumerate(sel_train_tmp):
            model.apply(lambda m: setattr(m, 'coder_channels', _coder_channels))
            model.apply(lambda m: setattr(m, 'en_stride', _en_stride))
            # print(model.coder_channels)
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()

            if index == len_train_tmp-1:
                loss =loss_function(pred, labels.to(device))
                loss.backward()
            else:
                loss =loss_function(pred, labels.to(device))
                loss.backward(retain_graph=True)
                accu_num -= torch.eq(pred_classes, labels.to(device)).sum()
            data_loader.set_description("[train epoch {}] {:18} loss: {:.3f}, acc: {:.3f}".format(epoch,str([_coder_channels, _en_stride]),
                                                                        accu_loss.item() / (step + 1),
                                                                        accu_num.item() / sample_num),refresh=True)
            
            # def f(m):
            #     if 'width_mult' in dir(m):
            #         print(m.width_mult)
            # model.apply(f)
            # data_loader.set_description("[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
            #                                                 accu_loss.item() / (step + 1),
            #                                                 accu_num.item() / sample_num),refresh=True)                                                           



        accu_loss += loss.detach()

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def train_one_epoch(model, optimizer, data_loader, device, epoch, display):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout,leave=display)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, display):
    loss_function = torch.nn.CrossEntropyLoss()
    model.apply(lambda m: setattr(m, 'width_mult', 1.0))
    model.apply(lambda m: setattr(m, 'quant_bits', 65535))
    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout,leave=display)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num



@torch.no_grad()
def sw_evaluate(model, data_loader, device, epoch, val_tmp):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    meters = {}
    res = {}
    val_tmp = tqdm(val_tmp, file=sys.stdout)
    for index,(_coder_channels, _en_stride) in enumerate(val_tmp):
        accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
        accu_loss = torch.zeros(1).to(device)  # 累计损失

        sample_num = 0
        # data_loader = tqdm(data_loader, file=sys.stdout)

        model.apply(lambda m: setattr(m, 'coder_channels', _coder_channels))
        model.apply(lambda m: setattr(m, 'en_stride', _en_stride))
        for step, data in enumerate(data_loader):
            images, labels = data
            sample_num += images.shape[0]

            pred = model(images.to(device))
            pred_classes = torch.max(pred, dim=1)[1]

            accu_num += torch.eq(pred_classes, labels.to(device)).sum()

            loss = loss_function(pred, labels.to(device))
            accu_loss += loss

            val_tmp.desc = "[valid epoch {}] {:18} loss: {:.3f}, acc: {:.3f}".format(epoch,str([_coder_channels, _en_stride]),
                                                                                accu_loss.item() / (step + 1),
                                                                                accu_num.item() / sample_num)


        res['loss'] = round(accu_loss.item() / (step + 1),3)
        res['acc'] = round(accu_num.item() / sample_num,3)
        # print(res)
        meters[str([_coder_channels,_en_stride])] = res.copy()

    return meters
    # return accu_loss.item() / (step + 1), accu_num.item() / sample_num
