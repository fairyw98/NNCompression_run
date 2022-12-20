import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from utils.my_dataset import MyDataSet

from models.mobilenet.sw_model_v2 import MobileNetV2 as create_model
from models.alexnet.model_unit import AlexNet as create_model
from utils.utils import Cluster, read_split_data, sw_evaluate, sw_train_one_epoch, evaluate

import csv

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # tb_writer = SummaryWriter()

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

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # model = create_model(num_classes=args.num_classes).to(device)
    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        # 删除有关分类类别的权重
        # for k in list(weights_dict.keys()):
        #     if "classifier" in k:
        #         del weights_dict[k]
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
    min_loss = 100.
    train_tmp = args.train_tmp
    val_tmp = args.val_tmp
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = sw_train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                train_tmp=train_tmp)

        # validate
        meters = sw_evaluate(model=model,
                            data_loader=val_loader,
                            device=device,
                            epoch=epoch,
                            val_tmp=val_tmp)
        print(meters)
        val_loss = meters[str(val_tmp[0])]['loss']
        val_acc = meters[str(val_tmp[0])]['acc']

        # tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        # tb_writer.add_scalar(tags[0], train_loss, epoch)
        # tb_writer.add_scalar(tags[1], train_acc, epoch)
        # tb_writer.add_scalar(tags[2], val_loss, epoch)
        # tb_writer.add_scalar(tags[3], val_acc, epoch)
        # tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # if val_acc > best_acc:
        #     best_acc = val_acc
        #     torch.save(model.state_dict(), "./weights/best_model.pth")

        # torch.save(model.state_dict(), "./weights/latest_model.pth")

        sum_acc = 0
        mean_best_acc = 0
        for key,value in meters.items():
            sum_acc += value['acc']
        mean_acc = sum_acc / len(meters.items())
        if mean_acc > mean_best_acc:
            mean_best_acc = mean_acc
            with open('sw_database.csv','w',newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['coder_channels','en_stride','min_loss','best_acc'])
                for key,value in meters.items():
                    param1 = key.split(',')[0].split('[')[-1]
                    param2 = key.split(',')[1].split(']')[0].split(' ')[-1]
                    # print(param2)
                    writer.writerow([param1,param2,value['loss'],value['acc']])

        # with open('sw_databse.txt','w') as f:
        #     f.write(str(meters))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    FLAGS_num_list = [8,4,16,32]
    FLAGS_quant_list = [1,2]
    train_tmp1 = [[i,j] for i in FLAGS_num_list for j in FLAGS_quant_list]
    train_tmp1 = [[8,1],[8,2],[4,1],[16,1],[32,2],[32,1]]
    train_tmp = []
    train_tmp.append(train_tmp1[-1])
    train_tmp = Cluster(train_tmp1)
    train_tmp = train_tmp1

    val_tmp = []
    val_tmp.append([FLAGS_num_list[-1], FLAGS_quant_list[-1]])
    val_tmp = train_tmp1
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--train_tmp', type=list, default=train_tmp)
    parser.add_argument('--val_tmp', type=list, default=val_tmp)
    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="./data_set/flower_data/flower_photos")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)

