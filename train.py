import os
from os.path import join
import glob
import inspect
from distutils.dir_util import copy_tree
import time
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim


from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from options import draw_process
from config import parse_args, parse_yacs
from config.const import PROJECT_ROOT, DATA_PATH, CSV_PATH
from data.get_dataloader import get_dataloader

from models.vgg16_gap import VGG16


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader, net, criterion, optimizer, epoch):
    pred_losses = AverageMeter()
    div_losses = AverageMeter()
    losses = AverageMeter()

    name_lst, perf_lst, pred_lst = [], [], []
    
    net.train()
    with tqdm(enumerate(train_loader), total= len(train_loader)) as pbar:
        for batch_num, (file_name, img, perf) in pbar:
            print(img.size())
            a = torch.randn(1,1,224,224)
            print(a.size())
            img = img.cuda()
            perf = perf.cuda()

            pred = net(img)

            loss = criterion(pred, perf)
            losses.update(loss.cpu().detach().numpy(), img.size(0))

            # 勾配の初期化
            optimizer.zero_grad()
            # 勾配の計算
            loss.backward()
            # 勾配を元にパラメータの更新
            optimizer.step()
            
            name_lst.extend(file_name)
            perf_lst.extend(perf.cpu().detach().numpy())
            pred_lst.extend(pred.cpu().detach().numpy())
            
            r2 = r2_score(perf_lst, pred_lst)
            
            pbar.set_postfix(Loss=losses.avg, R2=r2)

    log = OrderedDict([('loss', losses.avg), ('r2', r2), ('file_name', name_lst), ('perf', perf_lst), ('pred', pred_lst)])
    return log


def val(args, val_loader, net, criterion, epoch):
    pred_losses = AverageMeter()
    div_losses = AverageMeter()
    losses = AverageMeter()

    name_lst, perf_lst, pred_lst = [], [], []
    
    net.eval()
    with torch.no_grad():
        with tqdm(enumerate(val_loader), total= len(val_loader)) as pbar:
            for batch_num, (file_name, img, perf) in pbar:
                img = img.cuda()
                perf = perf.cuda()
                
                pred = net(img)

                loss = criterion(pred, perf)
                losses.update(loss.cpu().detach().numpy(), img.size(0))
                
                name_lst.extend(file_name)
                perf_lst.extend(perf.cpu().detach().numpy())
                pred_lst.extend(pred.cpu().detach().numpy())

                r2 = r2_score(perf_lst, pred_lst)

                pbar.set_postfix(Loss=losses.avg, R2=r2)

    log = OrderedDict([('loss', losses.avg), ('r2', r2), ('file_name', name_lst), ('perf', perf_lst), ('pred', pred_lst)])
    return log


def main():
    # 学習のオプションを読み込み
    # option = parse_args()
    args = parse_yacs()

    now = str(datetime.datetime.now())
    result_dir = join(PROJECT_ROOT, "result", now)
    # arg_file = join(result_dir, "args.yaml")
    # log_file = join(result_dir, "log.txt")

    # データの読み込み
    files = glob.glob(DATA_PATH)
    csv_file = pd.read_csv(CSV_PATH)

    # 交差検証(cross validation)
    kf = KFold(n_splits=args.TRAIN.KFOLD, shuffle=True, random_state=2020)

    for k, (train_index, val_index) in enumerate(kf.split(files)):
        # 交差検証用のフォルダ作成(remove)
        kfold_folder = join(result_dir, "kfold", f"k_{k+1}")
        if not os.path.exists(kfold_folder):
            os.makedirs(kfold_folder)

        train_files = np.array(files)[train_index].tolist()
        val_files = np.array(files)[val_index].tolist()

        train_files = train_files * 32

        # 元素ごとに解析 (remove)
        genso_lst = ['normal']

        for genso in genso_lst:
            # 計算結果用のフォルダ作成2 (remove)
            dir_path = join(kfold_folder, f"{genso}")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            # データの読み込み
            train_loader = get_dataloader(
                dataset=train_files, csv_file=csv_file,
                batch_size=args.TRAIN.BATCH_SIZE, type_dataset="train")
            val_loader = get_dataloader(
                dataset=val_files, csv_file=csv_file,
                batch_size=args.TRAIN.BATCH_SIZE, type_dataset="val")

            N_train = len(train_loader)
            N_val = len(val_loader)

            # ネットワークの読み込み
            net = VGG16(n_channels=1, n_classes=1)
            net = torch.nn.DataParallel(net).cuda()

            # モデルの読み込み (remove)
            """
            if not args.TRAIN.MODEL_TYPE == 'None':
                net.load_state_dict(torch.load(args.TRAIN.MODEL_TYPE))
                print('Model loaded from {}'.format(args.TRAIN.MODEL_TYPE))
            """

            criterion = nn.MSELoss().cuda()

            # 勾配法の設定
            optimizer = optim.SGD(
                net.parameters(),
                lr=args.TRAIN.LR,
                momentum=args.TRAIN.MOMENTUM,
                weight_decay=args.TRAIN.WEIGHT_DECAY)

            cudnn.benchmark = True

            learning_log = pd.DataFrame(index=[], columns=['epoch', 'lr', 'loss', 'r2', 'val_loss', 'val_r2'])

            best_acc = -100000000
            trigger = 0

            print('訓練開始:エポック:{} バッチサイズ:{} 学習率:{} 学習データ数:{} 検証データ数:{}'.format(
                args.TRAIN.EPOCHS, args.TRAIN.BATCH_SIZE, args.TRAIN.LR, N_train, N_val))
            print('CUDA:{}'.format(str(args.TRAIN.GPUS)))

            # エポック
            for epoch in range(1, args.TRAIN.EPOCHS+1):
                lr = args.TRAIN.LR * (0.5 ** ((epoch-1) // 25))
                adjust_learning_rate(optimizer, lr)

                since = time.time()
                print('*'*50)
                print('Strating epoch {}/{}.'.format(epoch, args.TRAIN.EPOCHS))
                print('*'*50)

                # 1エポック内での学習
                train_log = train(args, train_loader, net, criterion, optimizer, epoch)
                # 1エポック内での評価
                val_log = val(args, val_loader, net, criterion, epoch)

                tmp = pd.Series([
                    epoch,
                    lr,
                    train_log['loss'],
                    train_log['r2'],
                    val_log['loss'],
                    val_log['r2'],
                    ], index=['epoch', 'lr', 'loss', 'r2', 'val_loss', 'val_r2'])

                # ログの保存
                learning_log = learning_log.append(tmp, ignore_index=True)
                learning_log.to_csv(f'{dir_path}/log.csv', index=False)

                if args.DRAW_PROCESS:
                    draw_process(learning_log, f'{dir_path}/')

                trigger +=1

                print('='*50)
                print('Finished epoch {}/{}'.format(epoch, args.TRAIN.EPOCHS))
                print('--- loss:{:4f}  r2:{:4f} ---'.format(train_log['loss'], train_log['r2']))
                print('--- val_loss:{:4f}  val_r2:{:4f} ---'.format(val_log['loss'], val_log['r2']))

                # 最新推定結果のログ
                train_data_csv = pd.DataFrame({
                    'train_file':train_log['file_name'],
                    'train_perf':train_log['perf'],
                    'train_pred':train_log['pred']})
                train_data_csv.to_csv(f'{dir_path}/last_train_data.csv', index=False)

                val_data_csv = pd.DataFrame({
                    'val_file':val_log['file_name'],
                    'val_perf':val_log['perf'],
                    'val_pred':val_log['pred']})
                val_data_csv.to_csv(f'{dir_path}/last_val_data.csv', index=False)

                # 最良モデル/その時の推定結果の保存
                if val_log['r2'] > best_acc:
                    torch.save(net.module.state_dict(), f'{dir_path}/model.pth')
                    best_acc = val_log['r2']
                    print('BestModel saved!')
                    trigger = 0

                    train_data_csv = pd.DataFrame({
                        'train_file':train_log['file_name'],
                        'train_perf':train_log['perf'],
                        'train_pred':train_log['pred']})
                    train_data_csv.to_csv(f'{dir_path}/train_data.csv', index=False)

                    val_data_csv = pd.DataFrame({
                        'val_file':val_log['file_name'],
                        'val_perf':val_log['perf'],
                        'val_pred':val_log['pred']})
                    val_data_csv.to_csv(f'{dir_path}/val_data.csv', index=False)

                time_elapsed = time.time()-since
                print('エポック所要時間: {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
                print('='*50)

                torch.cuda.empty_cache()

                # Early stopping
                if not args.TRAIN.EARLY_STOP is None:
                    if trigger >= args.TRAIN.EARLY_STOP:
                        print('Early stopping!')
                        break

            print('Training finished!')

if __name__ == '__main__':
    main()
