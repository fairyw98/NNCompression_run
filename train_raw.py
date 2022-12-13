import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision

from utils.my_dataset import MyDataSet
from models.mobilenet.sw_model_v2 import MobileNetV2 as create_model
from models.vgg.model import vgg as create_model
from models.alexnet.nas_model import AlexNet as create_model
from models.vgg.nas_model import vgg as create_model
from models.vgg.model import vgg as create_model
from utils.utils import Cluster, colorstr, read_split_data, setup_seed, sw_evaluate, sw_train_one_epoch, evaluate, train_one_epoch

import utils.config as config

def main(args):
    # 设置随机数种子
    setup_seed(3407)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # if os.path.exists("./weights") is False:
    #     os.makedirs("./weights")

    # tb_writer = SummaryWriter()
    s = colorstr("Hello,it's fairy ! 😀 ")  # string
    print(s)
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # train_dataset = torchvision.datasets.CIFAR10(root='/home/wangfz/wksp/NNCompression_run/data_set/CIFAR10', 
    #                                             train = True, 
    #                                             transform = data_transform["train"], 
    #                                             download = False)
    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    # train_dataset = torchvision.datasets.CIFAR10(root='/home/wangfz/wksp/NNCompression_run/data_set/CIFAR10', 
    #                                             train = False, 
    #                                             transform = data_transform["val"], 
    #                                             download = False)
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # print('Using {} dataloader workers every process'.format(nw))
    print('Start training and validating...')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                            batch_size=batch_size,
    #                                            shuffle=True,
    #                                            pin_memory=True,
    #                                            num_workers=nw
    #                                            )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # val_loader = torch.utils.data.DataLoader(val_dataset,
    #                                          batch_size=batch_size,
    #                                          shuffle=False,
    #                                          pin_memory=True,
    #                                          num_workers=nw)
    model = create_model(num_classes=args.num_classes).to(device)
    # cfg = {}
    # cfg['coder_channels'] = args.coder_channels
    # cfg['en_stride'] = args.en_stride
    # # cfg['de_stride'] = args.de_stride
    # model = create_model(num_classes=args.num_classes,partition_id=args.partition_id,quant_bits=args.quantization,coder_cfg = cfg).to(device)
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "classifier" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "classifier" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1E-2)

    best_acc = 0.
    min_loss = 100
    # train_tmp = args.train_tmp
    # val_tmp = args.val_tmp
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                display = args.display)

        # validate
        val_loss,val_acc = evaluate(model=model,
                            data_loader=val_loader,
                            device=device,
                            epoch=epoch,
                            display = args.display)
        # print(meters)
        # val_loss = meters[str(val_tmp[0])]['loss']
        # val_acc = meters[str(val_tmp[0])]['acc']
        if args.tensorboard:
            tb_writer = SummaryWriter()
            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if val_acc > best_acc:
            best_acc = val_acc
        if val_loss < min_loss:
            min_loss = val_loss
        #     torch.save(model.state_dict(), f"./weights/p{args.partition_id}q{args.quantization}_best_model.pth")

        # torch.save(model.state_dict(), f"./weights/p{args.partition_id}q{args.quantization}_latest_model.pth")
    # if os.path.exists("database.csv") is True: 
    #     with open('database.csv','a+',newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow([args.partition_id,args.quantization,args.coder_channels,args.en_stride,best_acc])
    
    # config.best_acc = best_acc
    # config.min_loss = min_loss
    # return best_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="./data_set/flower_data/flower_photos")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='/home/wangfz/wksp/NNCompression_run/models/vgg/vgg16.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    parser.add_argument('--partition_id', type=int, default=3)
    parser.add_argument('--quantization', type=int, default=16)
    parser.add_argument('--coder_channels', type=int, default=16)
    parser.add_argument('--en_stride', type=int, default=2)
    parser.add_argument('--display', type=bool, default=True)
    parser.add_argument('--tensorboard', type=bool, default=True)
    opt = parser.parse_args()
    main(opt)

