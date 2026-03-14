#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import torch
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from torch import nn
import sys, time
CVPR26_ROOT = "/remote-home/share/25-jianfabai/cvpr2026"
sys.path.insert(0, CVPR26_ROOT)
import re
from utils import *
from tqdm import tqdm
from dataset1 import Lung3D_eccv_patient_supcon
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchnet import meter
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
from models.ResNet import SupConResNet
# import segmentation_models_pytorch as smp
# from efficientnet_pytorch_3d import EfficientNet3D
import pandas as pd
from collections import defaultdict
import torch.backends.cudnn as cudnn
import random
import math
import numpy
import glob

# torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
print("torch = {}".format(torch.__version__))
sys.stdout.reconfigure(line_buffering=True)

# torch.cuda.set_per_process_memory_fraction(0.99, device=0)  # 对GPU 0 设置95%的显存限制
# torch.cuda.set_per_process_memory_fraction(0.99, device=1)  # 对GPU 1 设置95%的显存限制
# torch.cuda.set_per_process_memory_fraction(0.99, device=2)  # 对GPU 0 设置95%的显存限制
# torch.cuda.set_per_process_memory_fraction(0.99, device=3)  # 对GPU 1 设置95%的显存限制


IMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

parser = argparse.ArgumentParser()
# parser.add_argument('--visname', '-vis', default='cvpr-pretrain-c2m1-mix-class-bai-v100-batch6-repeat', help='visname')
# parser.add_argument('--visname', '-vis', default='cvpr-pretrain-c1m1-nomix-class-bai-v100-batch6', help='visname')
parser.add_argument('--visname', '-vis', default='demo', help='visname')
parser.add_argument('--batch_size', '-bs', default=4, type=int, help='batch_size')
parser.add_argument('--lr', '-lr', default=1e-4, type=float, help='lr')  #####
parser.add_argument('--epochs', '-eps', default=100, type=int, help='epochs')
parser.add_argument('--n_classes', '-n_cls', default=2, type=int, help='n_classes')
parser.add_argument('--pretrain', '-pre', default=True, type=bool, help='use pretrained')
parser.add_argument('--supcon', '-con', default=False, type=bool, help='use supcon')
# parser.add_argument('--mixup', '-mix', default=True, type=bool, help='use mix')
parser.add_argument('--mixup', '-mix', default=False, type=bool, help='use mix')
parser.add_argument('--box_lung', '-box_lung', default=False, type=bool, help='data box lung')
parser.add_argument('--seg_sth', '-seg_something', default=None, type=str, help='lung or lesion, cat to input')

parser.add_argument('--iccv_test', '-iccv_test', default=False, type=bool, help='use iccv test as train')
parser.add_argument('--weighted_loss', '-wl', default=True, type=bool, help='weighted ce loss')
parser.add_argument('--mosmed', '-mm', default=False, type=bool, help='use mosmed in challenge 2')
parser.add_argument('--model', '-model', default='resnest50_3D', type=str, help='use mosmed in challenge 2')
parser.add_argument('--val_certain_epoch', '-val_certain_epoch', default=False, type=str, help='use mosmed in challenge 2')
parser.add_argument('--optimizer', '-optim', default='adam', type=str, help='use mosmed in challenge 2')


best_f1 = 0
val_epoch = 1
save_epoch = 10

TOPK = 2
SAVE_F1_TH = 0.95
best_ckpts = []  # list of tuples: (f1, filepath)



# random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# cudnn.deterministic = True

my_whole_seed = 0
torch.manual_seed(my_whole_seed)
torch.cuda.manual_seed_all(my_whole_seed)
torch.cuda.manual_seed(my_whole_seed)
np.random.seed(my_whole_seed)
random.seed(my_whole_seed)
cudnn.deterministic = True
cudnn.benchmark = False 


def scan_num(name: str):
    m = re.search(r'(\d+)$', name)
    return int(m.group(1)) if m else 10**18

def parse_args():
    global args
    args = parser.parse_args()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)/1048576

def try_save_topk(state, save_dir, epoch, f1, topk=2, best_ckpts=None):
    """
    Save checkpoint if it belongs to current top-k (by f1).
    Keep at most topk files on disk by removing the worst one.
    """
    assert best_ckpts is not None

    # 文件名带上 f1，方便人工查看
    save_name = os.path.join(save_dir, f"epoch{epoch:03d}_f1{f1:.4f}.pkl")

    # 先判断：如果现在还没满k，直接保存
    if len(best_ckpts) < topk:
        torch.save(state, save_name)
        best_ckpts.append((f1, save_name))
        best_ckpts.sort(key=lambda x: x[0], reverse=True)
        print(f"[TopK] Saved (not full): {save_name}")
        return True

    # 已满k：只有当 f1 > 当前最差的 f1 才更新
    worst_f1, worst_path = min(best_ckpts, key=lambda x: x[0])
    if f1 <= worst_f1:
        print(f"[TopK] Not saved (f1={f1:.4f} <= worst={worst_f1:.4f})")
        return False

    # 保存新的
    torch.save(state, save_name)
    best_ckpts.append((f1, save_name))
    best_ckpts.sort(key=lambda x: x[0], reverse=True)

    # 删除多出来的（删掉最差的）
    while len(best_ckpts) > topk:
        worst_f1, worst_path = best_ckpts.pop(-1)  # 最末尾就是最差
        if os.path.exists(worst_path):
            os.remove(worst_path)
            print(f"[TopK] Removed worst: {worst_path} (f1={worst_f1:.4f})")

    print(f"[TopK] Updated. Current kept: {[ (round(x[0],4), os.path.basename(x[1])) for x in best_ckpts ]}")
    return True


def main():
    print(torch.cuda.device_count())
    global best_f1
    global save_dir

    parse_args()
    if args.seg_sth:
        ipt_dim=2
    else:
        ipt_dim=1
    # prepare the model
    
    target_model = SupConResNet(name=args.model, ipt_dim=ipt_dim, head='mlp', feat_dim=128, n_classes=2, supcon=args.supcon)

    if args.supcon:
        s1 = target_model.sigma1
        s2 = target_model.sigma2



    if args.n_classes == 4:
        if args.model == 'P3DCResNet50' or args.model == 'medicalnet_resnet50':
            target_model.encoder.classifier = nn.Linear(2048,4)
        elif args.model == 'medicalnet_resnet34':
            target_model.encoder.classifier = nn.Linear(512,4)

        else:
            target_model.encoder.fc = nn.Linear(2048,4)
    
    if args.pretrain:
        # ckpt = torch.load('/remote-home/share/21-yuanruntian-21210240410/cmc/checkpoints/eccv/*clf_resnest50_con_mix_iccvtest/28.pkl')
        # V100上的torch为1.10.0时
        # ckpt = torch.load('./checkpoints/cvpr/pseudo-model12-mix/7.pkl')
        # 修改 3090上torch为2.系列时
        # ckpt = torch.load('./checkpoints/cvpr/pseudo-model12-mix/7.pkl', weights_only=False)
        # ckpt = torch.load('/remote-home/share/21-yuanruntian-21210240410/cmc/checkpoints/eccv/*clf_resnest50_con_mix_iccvtest/28.pkl', weights_only=False)
        # ckpt = torch.load('./checkpoints/cvpr/pseudo-model12-mix/7.pkl', weights_only=True)
        #c2m2
        # ckpt = torch.load('/remote-home/share/21-yuanruntian-21210240410/cmc/checkpoints/cvpr/pseudo-model12-mix-clf/1.pkl', weights_only=False)
        # ckpt = torch.load('/remote-home/share/21-yuanruntian-21210240410/cmc/checkpoints/cvpr/pseudo-model12-mix-clf/1.pkl')
        # c2m3
        # ckpt = torch.load('/remote-home/share/21-yuanruntian-21210240410/cmc/checkpoints/cvpr/pseudo-model12/14.pkl')
        # c2m4
        # ckpt = torch.load('/remote-home/share/21-yuanruntian-21210240410/cmc/checkpoints/cvpr/pseudo-model3-mix/9.pkl')
        # ckpt = torch.load('./checkpoints/cvpr/pseudo-model3-mix/9.pkl')
        # c2m5
        # ckpt = torch.load('/remote-home/share/21-yuanruntian-21210240410/cmc/checkpoints/cvpr/resnest50-noncon-nomix/3.pkl')
        # c1m1
        # ckpt = torch.load('/remote-home/share/21-yuanruntian-21210240410/da/checkpoints/71.pkl')
        # c2m4-b2-v100-38
        # ckpt = torch.load('/remote-home/share/25-jianfabai/cvpr2026/checkpoints/cvpr-pretrain-c2m4-nomix-class-bai-v100-batch6/epoch038_f10.9732.pkl')
        # ckpt = torch.load('/remote-home/share/25-jianfabai/cvpr2026/checkpoints/cvpr-pretrain-c2m4-nomix-class-bai-v100-batch6/epoch038_f10.9732.pkl', weights_only=False)
        # c2m4-b2-v100-52
        # ckpt = torch.load('/remote-home/share/25-jianfabai/cvpr2026/checkpoints/cvpr-pretrain-c2m4-nomix-class-bai-v100-batch6/epoch052_f10.9732.pkl')
        # ckpt = torch.load('/remote-home/share/25-jianfabai/cvpr2026/checkpoints/cvpr-pretrain-c2m4-nomix-class-bai-v100-batch6/epoch052_f10.9732.pkl', weights_only=False)
        # c2m5-b2-v100-5
        # ckpt = torch.load('/remote-home/share/25-jianfabai/cvpr2026/checkpoints/cvpr-pretrain-c2m5-nomix-class-bai-v100-batch6/epoch005_f10.9700.pkl')
        ckpt = torch.load('/remote-home/share/25-jianfabai/cvpr2026/checkpoints/cvpr-pretrain-c2m5-nomix-class-bai-v100-batch6/epoch005_f10.9700.pkl', weights_only=False)
        # c2m1-mix-b2-v100-21
        # ckpt = torch.load('/remote-home/share/25-jianfabai/cvpr2026/checkpoints/cvpr-pretrain-c2m1-mix-class-bai/epoch021_f10.9631.pkl')
        # ckpt = torch.load('/remote-home/share/25-jianfabai/cvpr2026/checkpoints/cvpr-pretrain-c2m1-mix-class-bai/epoch021_f10.9631.pkl', weights_only=False)
        # c2m1-nomix-b2-3090-1
        # ckpt = torch.load('/remote-home/share/25-jianfabai/cvpr2026/checkpoints/cvpr-pretrain-c2m1-nomix-class-bai/epoch001_f10.9665.pkl')
        # ckpt = torch.load('/remote-home/share/25-jianfabai/cvpr2026/checkpoints/cvpr-pretrain-c2m1-nomix-class-bai/epoch001_f10.9665.pkl', weights_only=False)
        # yuan-best-eccv-pretrain-mix/20.pkl
        # ckpt = torch.load('/remote-home/share/21-yuanruntian-21210240410/cmc/checkpoints/eccv-pretrain-mix/20.pkl', weights_only=False)
        state_dict = ckpt['net']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]

        target_model.load_state_dict(unParalled_state_dict, False)
        # print("iccv pretrain")

    print('Params: ', count_parameters(target_model))

    target_model = nn.DataParallel(target_model)
    target_model = target_model.cuda()
    
    # prepare data
    # train_data = Lung3D_eccv_patient_supcon(train=True,val=False,n_classes=args.n_classes, supcon=args.supcon, box_lung=args.box_lung, seg_sth=args.seg_sth, iccv_test=args.iccv_test, add_mosmed=args.mosmed)
    val_data = Lung3D_eccv_patient_supcon(train=False,val=True,n_classes=args.n_classes, supcon=args.supcon, box_lung=args.box_lung, seg_sth=args.seg_sth, iccv_test=args.iccv_test, add_mosmed=args.mosmed)

    # train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8,pin_memory=True,drop_last=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=8,pin_memory=True)

    #修改
    # train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,pin_memory=True,drop_last=True)
    # val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4,pin_memory=True)

    criterion = SupConLoss(temperature=0.1)
    criterion = criterion.cuda()
    if args.n_classes==4:
        if args.weighted_loss:
            if args.mosmed:
                weight = torch.tensor([0.0931, 0.0985, 0.1433, 0.6651]).cuda() #add mosmed
                # weight = torch.tensor([1., 1., 1., 2.]).cuda() #add mosmed
            else:
                weight = torch.tensor([0.1506, 0.2065, 0.1506, 0.4923]).cuda()
                # weight = torch.tensor([1., 1., 1., 2.]).cuda()
        else:
            weight=None
    else:
        weight = None
    print(weight)
    criterion_clf = nn.CrossEntropyLoss(weight=weight)
    criterion_clf = criterion_clf.cuda()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(target_model.parameters(), args.lr, weight_decay=1e-5)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(target_model.parameters(), args.lr, momentum=0.9, weight_decay=1e-5)


    con_matx = meter.ConfusionMeter(args.n_classes)

    save_dir = './checkpoints/'+ str(args.visname)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  
    test_log=open(save_dir+'/log.txt','w')   


    #先不用
    val1(target_model,val_loader,0,test_log, optimizer, save_dir)

         

@torch.no_grad()
def val1(net, val_loader, epoch,test_log, optimizer, save_dir):
    global best_f1
    parse_args()
    net = net.eval()

    correct = .0
    total = .0
    con_matx = meter.ConfusionMeter(args.n_classes)
    pred_list = []
    label_list = []

    # total_ = []
    # label_ = []
    all_ids = []
    pbar = tqdm(val_loader, ascii=True)

    for i, (data, masks, label, id) in enumerate(pbar):
        data = data.unsqueeze(1)
        # data = data.repeat(1,3,1,1,1)
        # print(id)

        data = data.float().cuda()
        label = label.float().cuda()
        if args.seg_sth:
            masks = masks.unsqueeze(1)
            masks = masks.float().cuda()
            data = torch.cat([data, masks], dim=1)

        _, feat, pred = net(data)

        # print(feat.size())
        # total_.append(feat)
        # label_.append(label)

        pred = F.softmax(pred)
        _, predicted = pred.max(1)

        pred_list.append(predicted.cpu().detach())
        label_list.append(label.cpu().detach())

        total += data.size(0)
        correct += predicted.eq(label.long()).sum().item()        
        con_matx.add(predicted.detach(),label.detach()) 
        pbar.set_description(' acc: %.3f'% (100.* correct / total))

        # 将 val 阶段的预测 flatten
        for i in id:
            all_ids.append(i)  # id 是样本名列表（如 ct_scan_0.npy）
    # ans = torch.cat(total_, dim=0)
    # ans = ans.cpu().numpy()

    # ans2 = torch.cat(label_, dim=0)
    # ans2 = ans2.cpu().numpy()
    # np.save('train_data.npy', ans)
    # np.save('train_label.npy', ans2)
    # print(ans.shape, ans2.shape)

    recall = recall_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)
    precision = precision_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)
    f1 = f1_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average='macro')
    f1_4 = f1_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)

    
    print(correct, total)
    acc = 100.* correct/total

    print('val epoch:', epoch, ' val acc: ', acc, 'recall:', recall, "precision:", precision, "f1_macro:",f1, 'f1:', f1_4)
    print(str(con_matx.value()))

    # 读取中心信息
    # df_covid = pd.read_csv('/remote-home/share/25-jianfabai/cvpr2026/validation_covid.csv')
    df_covid = pd.read_csv('/remote-home/share/25-jianfabai/cvpr2026/validation_covid1.csv')
    df_noncovid = pd.read_csv('/remote-home/share/25-jianfabai/cvpr2026/validation_non_covid.csv')

    # 两个独立映射，避免重名覆盖
    name2center_covid = dict(zip(df_covid['ct_scan_name'], df_covid['data_centre']))
    name2center_non   = dict(zip(df_noncovid['ct_scan_name'], df_noncovid['data_centre']))

    preds  = torch.cat(pred_list).numpy()
    labels = torch.cat(label_list).numpy()  # 这是 dataset 真值
    # all_ids 是 ct_scan_name 列表


    # 收集每个中心的样本（带名字）
    center_items = defaultdict(list)
    print("example all_ids:", all_ids[:5])


    for i, scan_name in enumerate(all_ids):
        true_lab = int(labels[i])
        pred_lab = int(preds[i])

        if true_lab == 1:
            center = name2center_covid.get(scan_name)
        else:
            center = name2center_non.get(scan_name)

        if center is None:
            continue

        center_items[center].append((scan_name, true_lab, pred_lab))

    # 计算每个中心的平均 F1
    center_f1_scores = {}
    # 打印前：对每个中心按 scan_num 排序
    for center in sorted(center_items.keys(), key=int):
        items = sorted(center_items[center], key=lambda x: scan_num(x[0]))
        y_true = [t for _, t, _ in items]
        y_pred = [p for _, _, p in items]
        names  = [n for n, _, _ in items]

        print(f"\n[{center}] first 10 ids: {names[:100]}")
        print(f"[{center}] True labels: {y_true[:100]} ... Pred labels: {y_pred[:100]} ... Total samples: {len(y_true)}")
        f1_covid = f1_score(y_true, y_pred, pos_label=1)
        f1_non = f1_score(y_true, y_pred, pos_label=0)
        if f1_covid and f1_non:
            avg_f1 = (f1_covid + f1_non) / 2
        else:
            avg_f1 = f1_non
        center_f1_scores[center] = avg_f1
        print(f"[{center}] F1_covid: {f1_covid:.4f}, F1_noncovid: {f1_non:.4f}, Avg: {avg_f1:.4f}")

    # 最终平均分数
    final_score = sum(center_f1_scores.values()) / len(center_f1_scores)
    print(f"\n>>> Final Averaged F1 Score Across Centers: {final_score:.4f}")

    test_log.write('Val Epoch:%d   Accuracy:%.4f   f1:%.4f   f1_avg:%.4f  con:%s \n'%(epoch,acc,f1,final_score, str(con_matx.value())))
    test_log.write(f'Per-center F1: {center_f1_scores}\n')
    test_log.flush() 

    #查看分类错误的
    wrong_all = []  # (center, scan_name, true, pred)
    for i, scan_name in enumerate(all_ids):
        t = int(labels[i])
        p = int(preds[i])
        if t != p:
            if t == 1:
                center = name2center_covid.get(scan_name)
            else:
                center = name2center_non.get(scan_name)
            wrong_all.append((center, scan_name, t, p))

    print("\n========== Misclassified samples (Global) ==========")
    print(f"Total wrong: {len(wrong_all)} / {len(all_ids)}")
    for center, scan_name, t, p in wrong_all:
        print(f"center={center}  id={scan_name}  true={t}  pred={p}")

    # 按 center 分组打印
    # from collections import defaultdict
    wrong_by_center = defaultdict(list)
    for center, scan_name, t, p in wrong_all:
        wrong_by_center[center].append((scan_name, t, p))

    def _center_sort_key(x):
        s = str(x)
        # 纯数字（包括 "0","1"...）按数字排
        if s.isdigit():
            return (0, int(s))
        # 其他按字符串排
        return (1, s)

    print("\n========== Misclassified samples (Per-center) ==========")
    # for center in sorted(wrong_by_center.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
    for center in sorted(wrong_by_center.keys(), key=_center_sort_key):
        items = wrong_by_center[center]
        print(f"\n[center {center}] wrong {len(items)} samples:")
        # items = sorted(items, key=lambda x: scan_num(x[0]))
        for scan_name, t, p in items:
            print(f"id={scan_name}  true={t}  pred={p}")

    # exit(0)

    global best_f1, best_ckpts

    if not args.val_certain_epoch:
        if f1 >= SAVE_F1_TH:
            state = {
                'net': net.state_dict(),
                'f1': f1,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            saved = try_save_topk(
                state=state,
                save_dir=save_dir,
                epoch=epoch,
                # f1=f1,
                f1 = final_score,
                topk=TOPK,
                best_ckpts=best_ckpts
            )
            if saved:
                best_f1 = max(best_f1, f1)



if __name__ == "__main__":
    t0 = time.time()
    main()
    t1 = time.time()
    elapsed = t1 - t0
    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60
    print(f"elapsed={int(hours)}h {int(minutes)}m {seconds:.3f}s")
        

