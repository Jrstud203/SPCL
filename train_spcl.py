import argparse
import copy
import logging
import os

from torch import nn

from dgcl.dataset import get_cardiac_loaders, get_acdcal_loaders, get_scd_loaders
from dgcl.utils.model_init import init_weight

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os.path as osp
import pprint
import random
import sys
import time
import warnings
from datetime import datetime
from itertools import cycle
from torchvision.utils import make_grid

import numpy as np
import torch
# cpu_num = 6  # 这里设置成你想运行的CPU个数
# os.environ["OMP_NUM_THREADS"] = str(cpu_num)  # noqa
# os.environ["MKL_NUM_THREADS"] = str(cpu_num) # noqa
# torch.set_num_threads(cpu_num)
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter

from dgcl.dataset.augmentation import generate_unsup_data
from dgcl.models.model_helper import ModelBuilder
from dgcl.metrics import *
from dgcl.utils.loss_helper import (
    get_criterion, DiceLoss
)
from dgcl.utils.lr_helper import get_optimizer, get_scheduler
from dgcl.utils.utils import (
    AverageMeter,
    init_log,
    intersectionAndUnion,
    load_state,
    set_random_seed, init_logger, sigmoid_rampup, RampdownScheduler,
)

from torch.cuda.amp import GradScaler, autocast
from dgcl.dataset.acdcal_dataset import get_acdcal_loaders
from skimage.segmentation import find_boundaries


from dgcl.bcl.spcl import (
    FeatureMemory, compute_spcl_loss,
)
from tqdm import tqdm
from scipy.ndimage.morphology import distance_transform_edt
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
# acdc_config.yaml
# mm_config.yaml
# promise.yaml
parser.add_argument("--config", type=str, default="/configs/spcl/scd3.yaml")
parser.add_argument('--exp', type=str, default=f'SCD/spcl', help='the dir to save logs and models')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training. default=1')
parser.add_argument("--seed", type=int, default=1337, help='default=0')
parser.add_argument("--num_anchor", type=int, default=256, help='anchor number for each subclass. default=200')
parser.add_argument("--num_bank", type=int, default=400, help='bank size for each subclass. default=400')
parser.add_argument("--contra_weight", type=float, default=0.1, help='default=0.5')
parser.add_argument("--max_bw", type=float, default=0.1, help='default=0.1')
parser.add_argument("--confi_an", type=float, default=0.75, help='prob threshold for hard anchor')
parser.add_argument('--b_update', type=str, default='mean', help='update scheme of bound. default=mean/prob/prob_p')
parser.add_argument("--confi_b", type=float, default=0.85, help='prob threshold of represent bound')
parser.add_argument("--radius", type=int, default=0, help="to split regions. 0,2,4")
parser.add_argument("--temp_i", type=float, default=0.5, help="temp for inner")
parser.add_argument("--temp_b", type=float, default=0.07, help="temp for bound")
# dgcl origin
parser.add_argument("--alpha_t", type=float, default=80, help='for rep space')
parser.add_argument("--un_weight", type=float, default=0.3, help='default=0.5')
parser.add_argument("--conf_threshold", type=float, default=0.95, help="confidence threshold for using pseudo-labels")

best_stu_prec = 0
best_ema_prec = 0

def set_deterministic(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False  # address: Unable to find a valid cuDNN algorithm to run convolution
        torch.backends.cudnn.benchmark = False


def _random_rotate(image, label, prior):
    angle = float(torch.empty(1).uniform_(-20., 20.).item())
    image = TF.rotate(image, angle)
    label = TF.rotate(label, angle)
    prior = TF.rotate(prior, angle)
    return image, label, prior


def get_pixel_sets_distrans(src_sets, radius=2):
    """
        src_sets: shape->[N, 28, 28]
    """
    if isinstance(src_sets, torch.Tensor):
        src_sets = src_sets.numpy()
    if isinstance(src_sets, np.ndarray):
        keeps = []
        for src_set in src_sets:
            # 计算数组中值为 非零 的点到最近 零值点 的距离，并同时返回距离矩阵
            # 计算每个点到边界的距离， Ture为 inn and out
            keep = distance_transform_edt(np.logical_not(src_set))
            keep = keep < radius
            keeps.append(keep.astype(np.float32))
    else:
        raise ValueError(f'only np.ndarray is supported!')
    return torch.tensor(keeps).to(dtype=torch.long)


def main():
    global args, cfg
    args = parser.parse_args()
    set_deterministic(args.seed)

    def worker_init_fn(worker_id):
        random.seed(worker_id + args.seed)

    cfg = yaml.load(open(sys.path[0] + args.config, "r"), Loader=yaml.Loader)

    log_dir = "../model/{}_{}/{}".format(args.exp, cfg["dataset"]["kwargs"]['labeled_num'], 'deeplabv3+')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = init_logger("global", True, log_dir, logging.INFO, 'w')
    logger.propagate = 0

    logger.info("{}".format(pprint.pformat(cfg)))
    logger.info(f'\n{args}\n')
    logger.info(f"path of log: {log_dir}")
    tb_logger = SummaryWriter(log_dir)

    model = ModelBuilder(cfg["net"])
    modules_back = [model.encoder]
    if cfg["net"].get("aux_loss", False):
        modules_head = [model.auxor, model.decoder]
    else:
        modules_head = [model.decoder]

    model.cuda()
    # 增加模型权重初始化
    init_weight(model.decoder, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-5, 0.1,
                mode='fan_in', nonlinearity='relu')

    sup_loss_fn = get_criterion(cfg)
    dice_loss = DiceLoss(n_classes=cfg["net"]["num_classes"], ignore_index=cfg["ignore_label"])

    cfg_dset = cfg["dataset"]["kwargs"]
    cfg_dset['worker_init_fn'] = worker_init_fn
    if "acdcal" in cfg["dataset"]["name"]:
        get_loader = get_acdcal_loaders
    elif "cardiac" in cfg["dataset"]["name"]:
        get_loader = get_cardiac_loaders
    elif "scd" in cfg["dataset"]["name"]:
        get_loader = get_scd_loaders
    else:
        raise ValueError
    train_loader, val_loader = get_loader(**cfg_dset)

    # 构造评估dice and jaccard
    metrics_dcja = []
    for metric_name, metric_cfg in cfg['metrics'].items():
        metrics_dcja.append(eval(f"{metric_name}")(**metric_cfg))

    # Optimizer and lr decay scheduler
    cfg_trainer = cfg["trainer"]
    cfg_trainer["epochs"] = cfg_trainer['max_iter'] // len(train_loader) + 1
    cfg["trainer"]["epochs"] = cfg_trainer["epochs"]

    cfg_optim = cfg_trainer["optimizer"]
    times = 1  # 1->10
    params_list = []
    for module in modules_back:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )
    for module in modules_head:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * times)
        )

    optimizer = get_optimizer(params_list, cfg_optim)

    # Teacher model
    model_teacher = ModelBuilder(cfg["net"])
    model_teacher = model_teacher.cuda()
    # 增加模型权重初始化
    init_weight(model_teacher.decoder, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-5, 0.1,
                mode='fan_in', nonlinearity='relu')

    for p in model_teacher.parameters():
        p.requires_grad = False

    last_epoch = 0 #############################

    # auto_resume > pretrain
    if cfg["saver"].get("auto_resume", False):
        lastest_model = os.path.join(cfg["save_path"], "ckpt_ori.pth")
        if not os.path.exists(lastest_model):
            "No checkpoint found in '{}'".format(lastest_model)
        else:
            print(f"Resume model from: '{lastest_model}'")
            best_prec, last_epoch = load_state(
                lastest_model, model, optimizer=optimizer, key="model_state"
            )
            _, _ = load_state(
                lastest_model, model_teacher, optimizer=optimizer, key="teacher_state"
            )

    elif cfg["saver"].get("pretrain", False):
        load_state(cfg["saver"]["pretrain"], model, key="model_state")
        load_state(cfg["saver"]["pretrain"], model_teacher, key="teacher_state")

    optimizer_start = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(
        cfg_trainer, len(train_loader), optimizer_start, start_epoch=last_epoch
    )

    # build class-wise memory bank
    memobank = FeatureMemory(memory_per_class=args.num_bank, n_classes=cfg["net"]["num_classes"], b_update=args.b_update)

    # Start to train model
    iterator = tqdm(range(last_epoch, cfg_trainer["epochs"]), ncols=70)
    for epoch in iterator:
        # Training
        train(
            model,
            model_teacher,
            optimizer,
            lr_scheduler,
            sup_loss_fn,
            dice_loss,
            train_loader,
            val_loader,
            metrics_dcja,
            epoch,
            tb_logger,
            logger,
            memobank,
            log_dir,
        )

        if (len(train_loader) * (epoch + 1)) >= cfg_trainer['max_iter']:
            break

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

    iterator.close()
    logger.info('best_stu_prec: {:.4f}, best_ema_prec: {:.4f}'.format(best_stu_prec, best_ema_prec))
    return "Training Finished!"


def train(
    model,
    model_teacher,
    optimizer,
    lr_scheduler,
    sup_loss_fn,
    dice_loss,
    train_loader,
    val_loader,
    metrics_dcja,
    epoch,
    tb_logger,
    logger,
    memobank,
    log_dir,
):
    global best_stu_prec, best_ema_prec
    ema_decay_origin = cfg["net"]["ema_decay"]
    max_iter = cfg["trainer"]["max_iter"]
    model.train()

    sup_losses = AverageMeter(10)
    uns_losses = AverageMeter(10)
    con_losses = AverageMeter(10)
    icon_losses = AverageMeter(10)
    bcon_losses = AverageMeter(10)
    data_times = AverageMeter(10)
    batch_times = AverageMeter(10)
    learning_rates = AverageMeter(10)

    batch_end = time.time()
    for step, sampled_batch in enumerate(train_loader):
        batch_start = time.time()
        data_times.update(batch_start - batch_end)

        i_iter = epoch * len(train_loader) + step
        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step()

        labeled_bs = cfg['dataset']['kwargs']['labeled_bs']
        image_, label_, ignore_ = sampled_batch

        # 进行旋转
        images_, labels_, ignores_ = [], [], []
        for image_i, label_i, ignore_i in zip(image_, label_, ignore_):
            image_i, label_i, ignore_i = _random_rotate(image_i, label_i.unsqueeze(0), ignore_i.unsqueeze(0))
            images_.append(image_i)
            labels_.append(label_i)
            ignores_.append(ignore_i)
        image_ = torch.stack(images_, dim=0)
        label_ = torch.stack(labels_, dim=0).squeeze(1)
        ignore_ = torch.stack(ignores_, dim=0).squeeze(1)

        image_l, label_l = image_[:labeled_bs], label_[:labeled_bs]  # b3hw, bhw
        _, h, w = label_l.size()
        image_l, label_l = image_l.cuda(), label_l.cuda()

        image_u, label_u, ignore_mask = image_[labeled_bs:], label_[labeled_bs:], ignore_[labeled_bs:]
        image_u, label_u = image_u.cuda(), label_u.cuda()
        ignore_mask = ignore_mask.cuda()

        # unsupervised loss
        pre_epoch = cfg["trainer"].get("sup_only_epoch", 1)
        total_epoch = cfg["trainer"]["epochs"]
        progress = (epoch-pre_epoch) / (total_epoch-pre_epoch)

        alpha_t = max(args.alpha_t * (1 - progress), 10)  # from 80 to 10    用于输出维度大小上的

        num_anchor = args.num_anchor
        contra_weight = args.contra_weight

        if epoch < cfg["trainer"].get("sup_only_epoch", 1):
            # forward
            outs = model(image_l)
            pred, rep = outs["pred"], outs["rep"]
            pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)

            # supervised loss
            if "aux_loss" in cfg["net"].keys():
                aux = outs["aux"]
                aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
                sup_loss = sup_loss_fn([pred, aux], label_l)
            else:
                sup_loss = 0.5 * (sup_loss_fn(pred, label_l) +
                                  dice_loss(torch.softmax(pred, dim=1), label_l.clone().unsqueeze(1))
                                  )

            model_teacher.train()
            _ = model_teacher(image_l)

            unsup_loss = 0 * rep.sum()
            contra_loss = 0 * rep.sum()
            contrast_loss = [torch.tensor(0.0), torch.tensor(0.0)]

            loss = sup_loss + unsup_loss + contra_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            if epoch == cfg["trainer"].get("sup_only_epoch", 1)  and step == 0:
            # copy student parameters to teacher
                with torch.no_grad():
                    for t_params, s_params in zip(
                        model_teacher.parameters(), model.parameters()
                    ):
                        t_params.data = s_params.data

            num_labeled = len(image_l)
            num_unlabeled = len(image_u)
            image_bi = torch.cat((image_l, image_u))
    
            """构建标签维度的伪标签"""
            model_teacher.train()
            with torch.no_grad():
                out_bi_t = model_teacher(image_bi)
                pred_bi_t = out_bi_t["pred"].detach()
                
                # Get predictions of original unlabeled data
                pred_u_t = pred_bi_t[num_labeled:]
                
                pred_u_t_large = F.interpolate(pred_u_t, (h, w), mode="bilinear", align_corners=True)
                prob_u_t_large = F.softmax(pred_u_t_large, dim=1)
                logits_u_t, label_u_t = torch.max(prob_u_t_large, dim=1)

                # fix-match ops
                thresh_mask = logits_u_t < args.conf_threshold
                label_u_t[thresh_mask] = 255

                if random.uniform(0,1) < 0.5:  # 0.5
                    image_u_aug, label_u_aug, ignore_mask_aug = generate_unsup_data(
                        image_u,
                        label_u_t.clone(),
                        ignore_mask.clone(),
                        mode='cutmix',
                    )
                else:
                    image_u_aug, label_u_aug, ignore_mask_aug = image_u, label_u_t.clone(), ignore_mask.clone()

            image_tri = torch.cat((image_l, image_u, image_u_aug), dim=0)
            out_tri_s = model(image_tri)
            pred_tri_s, rep_tri_s, fts_tri_s = out_tri_s["pred"], out_tri_s["rep"], out_tri_s['fts']

            # 用于anchor的采样
            pseudo_bi_s_logits_cls = torch.max(torch.softmax(pred_tri_s, dim=1), dim=1)[0][:num_labeled+num_unlabeled]

            pred_tri_s_large = F.interpolate(pred_tri_s, size=(h, w), mode="bilinear", align_corners=True)

            pred_l_s_large = pred_tri_s_large[:num_labeled]
            pred_u_aug_s_large = pred_tri_s_large[num_labeled+num_unlabeled:]

            # supervised loss
            if "aux_loss" in cfg["net"].keys():
                aux = out_tri_s["aux"][:num_labeled]
                aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
                sup_loss = sup_loss_fn([pred_l_s_large, aux], label_l.clone())
            else:
                sup_loss = 0.5 * (sup_loss_fn(pred_l_s_large, label_l.clone()) +
                                  dice_loss(torch.softmax(pred_l_s_large, dim=1), label_l.clone().unsqueeze(1))
                                  )

            unsup_loss = dice_loss(torch.softmax(pred_u_aug_s_large, dim=1), label_u_aug.clone().unsqueeze(1))

            # contrastive loss using pseudo labels
            with torch.no_grad():
                # pred_bi_t 
                probs_bi_t = torch.softmax(pred_bi_t, dim=1) 
                _, probs_u_t = probs_bi_t[:num_labeled], probs_bi_t[num_labeled:]

                label_l_small = F.interpolate(label_l.unsqueeze(1).float(),size=pred_bi_t.shape[2:],mode="nearest").squeeze(1)
                _, label_u_small = torch.max(probs_u_t, dim=1)
                ignore_mask_small = F.interpolate(ignore_mask.unsqueeze(1).float(),size=pred_bi_t.shape[2:],mode="nearest").squeeze(1).long()
                label_info = torch.cat([label_l_small.long(), label_u_small.detach().clone().long()], dim=0)

                # 根据entropy确定可靠的输出维度的伪标签
                entropy_u = -torch.sum(probs_u_t * torch.log(probs_u_t + 1e-10), dim=1)
                # 一开始获取20%的伪标签到后来的90%的伪标签
                high_thresh_u = torch.quantile(entropy_u[ignore_mask_small != 255].detach().flatten(), (100 - alpha_t)/100)
                high_entropy_mask_u = (entropy_u.ge(high_thresh_u).bool() * (ignore_mask_small != 255).bool())
                label_u_small[high_entropy_mask_u] = 255        
                label_u_small[ignore_mask_small == 255] = 255 

                # The important label
                label_contra_memo = torch.cat([label_l_small.long(), label_u_small.long()], dim=0)

            rep_bi_s = rep_tri_s[:num_labeled+num_unlabeled]
            # 确定边界
            if label_info is not None:
                bound = []
                for mi in range(label_info.shape[0]):
                    bound_bool = find_boundaries(label_info[mi, :, :].cpu().numpy().astype(np.int8), mode='thick')
                    bound.append(torch.from_numpy(bound_bool))
                bound = torch.stack(bound, dim=0).int().cuda()  # bsz,h,w
                if args.radius > 0:
                    # 扩展边界和去边界
                    edge_sets_dilate = get_pixel_sets_distrans(bound.cpu(), radius=args.radius)
                    bound = edge_sets_dilate.cuda() - bound
            else:
                print('during computing bound, label_info should be provided!')
                raise Exception

            if memobank.check_if_full():
                contrast_loss = compute_spcl_loss(rep=rep_bi_s, memo=memobank, label=label_contra_memo.detach(),
                                                  bound=bound.detach(), prob=pseudo_bi_s_logits_cls,
                                                  temperature=[args.temp_i, args.temp_b], k_low_thresh=num_anchor,
                                                  strong_threshold=args.confi_an)
                if contrast_loss is None:
                    contra_loss = 0*rep_tri_s.sum()
                    contrast_loss = [torch.tensor(0.0), torch.tensor(0.0)]
                else:
                    factor = args.max_bw * sigmoid_rampup(i_iter // 100, 60.0)
                    contra_loss = (1 - factor) * contrast_loss[0] + factor * contrast_loss[1]
            else:
                contra_loss = 0*rep_tri_s.sum()
                contrast_loss = [torch.tensor(0.0), torch.tensor(0.0)]

            loss = sup_loss + args.un_weight * unsup_loss + contra_weight * contra_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update memory bank
            p2mask = None
            if args.b_update == 'prob' or args.b_update == 'prob_p':
                pred_bi_s = torch.softmax(pred_tri_s[:num_labeled+num_unlabeled], dim=1)
                prob_bi_s, pred_bi_s = torch.max(pred_bi_s, dim=1)
                p2mask = (pred_bi_s == label_contra_memo) * (prob_bi_s > args.confi_b)
            memobank.update(rep_bi_s.float().detach(), label_contra_memo.detach(), bound.detach(), p2mask)

            with torch.no_grad():
                ema_decay = min(
                    1
                    - 1
                    / (
                        i_iter
                        - len(train_loader) * cfg["trainer"].get("sup_only_epoch", 1)
                        + 1
                    ),
                    ema_decay_origin,
                )

                for t_params, s_params in zip(
                    model_teacher.parameters(), model.parameters()
                ):
                    t_params.data = (
                        ema_decay * t_params.data + (1 - ema_decay) * s_params.data
                    )

        sup_losses.update(sup_loss.item())
        uns_losses.update(unsup_loss.item())
        con_losses.update(contra_loss.item())
        icon_losses.update(contrast_loss[0].item())
        bcon_losses.update(contrast_loss[1].item())

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)

        if (i_iter+1) % 50 == 0:
            logger.info(
                "Iter [{}/{}]\t"
                "con_w {:.3f}\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Sup {sup_loss.val:.3f} ({sup_loss.avg:.3f})\t"
                "Uns {uns_loss.val:.3f} ({uns_loss.avg:.3f})\t"
                "Con {con_loss.val:.3f} ({con_loss.avg:.3f})\t"
                "iCon {icon_loss.val:.3f} ({icon_loss.avg:.3f})\t"
                "bCon {bcon_loss.val:.3f} ({bcon_loss.avg:.3f})\t".format(
                    i_iter+1,
                    max_iter,
                    contra_weight,
                    batch_time=batch_times,
                    sup_loss=sup_losses,
                    uns_loss=uns_losses,
                    con_loss=con_losses,
                    icon_loss=icon_losses,
                    bcon_loss=bcon_losses,
                )
            )

            tb_logger.add_scalar("train/lr", learning_rates.val, i_iter)
            tb_logger.add_scalar("train/Sup Loss", sup_losses.val, i_iter)
            tb_logger.add_scalar("train/Uns Loss", uns_losses.val, i_iter)
            tb_logger.add_scalar("train/Con Loss", con_losses.val, i_iter)
            tb_logger.add_scalar("train/iCon Loss", icon_losses.val, i_iter)
            tb_logger.add_scalar("train/bCon Loss", bcon_losses.val, i_iter)

            # 保存图片
            if epoch >= cfg["trainer"].get("sup_only_epoch", 1):
                tb_logger.add_image("train/images", make_grid(image_bi, 4, normalize=True), i_iter+1)

                pred_bi_t = out_bi_t["pred"].detach()
                pred_t = F.interpolate(pred_bi_t.clone(), (h, w), mode="bilinear", align_corners=True)
                pred_t = torch.argmax(pred_t, dim=1, keepdim=True).to(torch.float)
                tb_logger.add_image("train/pred_t", make_grid(pred_t * 50., 4, normalize=False), i_iter+1)

                pred_s = pred_tri_s_large[:num_labeled + num_unlabeled]
                pred_s = torch.argmax(pred_s.detach().clone(), dim=1, keepdim=True).to(torch.float)
                tb_logger.add_image("train/pred_s", make_grid(pred_s * 50., 4, normalize=False), i_iter+1)

                label_bi = torch.cat([label_l.unsqueeze(1), label_u.unsqueeze(1)], dim=0)
                tb_logger.add_image("train/labels", make_grid(label_bi * 50., 4, normalize=False), i_iter+1)

                bound_bi = bound.unsqueeze(1)
                tb_logger.add_image("train/bound", make_grid(bound_bi, 4, normalize=False), i_iter + 1)

        if (i_iter+1) % 1000 == 0:
            model.eval()
            logger.info(f"in iter {i_iter+1}, evaluation")
            val_res, _, val_table = validate(model, val_loader, metrics_dcja)
            logger.info(f'val_s result:\n{val_table.get_string()}')

            if val_res['Dice']['Mean'] > best_stu_prec:
                best_stu_prec = val_res['Dice']['Mean']
                save_mode_path = os.path.join(log_dir, 'stu_best.pth')
                torch.save(model.state_dict(), save_mode_path)

            model.train()

            if epoch >= cfg["trainer"].get("sup_only_epoch", 1):
                model_teacher.eval()
                val_res, _, val_table = validate(model_teacher, val_loader, metrics_dcja)
                logger.info(f'val_t result:\n{val_table.get_string()}')

                if val_res['Dice']['Mean'] > best_ema_prec:
                    best_ema_prec = val_res['Dice']['Mean']
                    save_mode_path = os.path.join(log_dir, 'ema_best.pth')
                    torch.save(model_teacher.state_dict(), save_mode_path)

                model_teacher.train()

        if (i_iter+1) % max_iter == 0:
            state = {
                "iter": i_iter + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "teacher_state": model_teacher.state_dict(),
            }
            torch.save(state, osp.join(log_dir, 'iter_' + str(i_iter+1) + '.pth'))

        if (i_iter+1) >= max_iter:
            break


def validate(
    model,
    data_loader,
    metrics_dcja,
):
    model.eval()
    val_res = None
    val_scalars = {}

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["ignore_label"],
    )

    ig_num = 0
    for step, batch in enumerate(data_loader):
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()
        if len(labels.unique()) == 1:
            print(f"{step} val scan only contain background")
            ig_num += 1
            continue

        with torch.no_grad():
            outs = model(images)

        # get the output produced by model_teacher
        output = outs["pred"]
        output = F.interpolate(
            output, labels.shape[1:], mode="bilinear", align_corners=True
        )
        output = output.data.max(1)[1]
        target_origin = labels  # B,H,W
        output[target_origin == ignore_label] = ignore_label

        batch_res = {}
        for metric_i in metrics_dcja:
            batch_res[metric_i.name] = metric_i(output.unsqueeze(1), target_origin.unsqueeze(1))

        if val_res is None:
            val_res = batch_res
        else:
            for metric_name in val_res.keys():
                for key in val_res[metric_name].keys():
                    val_res[metric_name][key] += batch_res[metric_name][key]

    for metric_name in val_res.keys():
        for key in val_res[metric_name].keys():
            val_res[metric_name][key] = val_res[metric_name][key] / (len(data_loader) - ig_num)
            val_scalars[f'val/{metric_name}.{key}'] = val_res[metric_name][key]

        val_res_list = [_.cpu() for _ in val_res[metric_name].values()]
        # don't consider the background
        val_res[metric_name]['Mean'] = np.mean(val_res_list[1:])
        val_scalars[f'val/{metric_name}.Mean'] = val_res[metric_name]['Mean']

    val_table = PrettyTable()
    val_table.field_names = ['Metirc'] + list(list(val_res.values())[0].keys())
    for metric_name in val_res.keys():
        if metric_name in ['Dice', 'Jaccard', 'Acc', 'IoU', 'Recall', 'Precision']:
            temp = [float(format(_ * 100, '.2f')) for _ in val_res[metric_name].values()]
        else:
            temp = [float(format(_, '.2f')) for _ in val_res[metric_name].values()]
        val_table.add_row([metric_name] + temp)

    return val_res, val_scalars, val_table


if __name__ == "__main__":
    main()
