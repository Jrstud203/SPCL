from random import sample
import torch
import torch.nn.functional as F
import numpy as np


class FeatureMemory:
    def __init__(self, memory_per_class=10000, n_classes=21, proj_dim=256, b_update="mean"):
        self.memory_per_class = memory_per_class
        self.n_classes = n_classes
        self.proj_dim = proj_dim
        self.proj_memory = [[], []]
        self.b_update = b_update

        for i in range(self.n_classes):
            self.proj_memory[0].append(torch.zeros(0, proj_dim).cuda())  # Memo for inner rep
            self.proj_memory[1].append(torch.zeros(0, proj_dim).cuda())  # Memo for bound rep

    def check_if_full(self):
        full = True
        for type_list in self.proj_memory:
            for item in type_list:  # self.fts_memory
                if item.size(0) < self.memory_per_class:
                    full = False
        return full

    def concat_memory(self):
        """in segment queue, obtain all segment pixel features from all classes"""
        n_view = 2
        x_ = torch.zeros((self.n_classes * self.memory_per_class, n_view, self.proj_dim)).float().cuda()
        y_ = torch.zeros((self.n_classes * self.memory_per_class, n_view)).float().cuda()

        sample_ptr = 0
        for c in range(self.n_classes):
            x_[sample_ptr:sample_ptr + self.memory_per_class, 0, :] = self.proj_memory[0][c]
            x_[sample_ptr:sample_ptr + self.memory_per_class, 1, :] = self.proj_memory[1][c]
            y_[sample_ptr:sample_ptr + self.memory_per_class, ...] = c
            sample_ptr += self.memory_per_class
        return x_, y_

    @torch.no_grad()
    def update(self, rep, labels, bound, p2mask=None):
        # rep = rep.permute(0,2,3,1) # B H W C
        feat_inner, feat_bound = [], []
        for cls in range(self.n_classes):
            feat_inner.append([])
            feat_bound.append([])

        for bs in range(rep.shape[0]):
            this_feat = rep[bs].contiguous().view(self.proj_dim, -1)
            this_label = labels[bs].contiguous().view(-1)
            this_bound = bound[bs].contiguous().view(-1)
            if p2mask is not None:
                this_p2mask = p2mask[bs].contiguous().view(-1)
            batch_cls = torch.unique(this_label)
            cls_filt = (batch_cls != 255)
            batch_cls = batch_cls[cls_filt]

            # only save the mean segment feature from the same class and the same image
            for lb in batch_cls:
                idxs = (this_label == lb).nonzero()
                lb = int(lb.item())
                inner_idxs = idxs[this_bound[idxs] == 0.0]
                bound_idxs = idxs[this_bound[idxs] == 1.0]
                # segment enqueue and dequeue
                proto = None
                if len(inner_idxs) > 0.0:
                    # 计算平均特征
                    feat = torch.mean(this_feat[:, inner_idxs], dim=1).squeeze()
                    proto = F.normalize(feat.view(-1), p=2, dim=0)
                    feat_inner[lb].append(proto)

                if len(bound_idxs) > 0.0:
                    if self.b_update == "mean":
                        # 计算平均特征
                        feat = torch.mean(this_feat[:, bound_idxs], dim=1).squeeze()
                        feat_bound[lb].append(F.normalize(feat.view(-1), p=2, dim=0))
                    elif self.b_update == "rand":
                        num_pixel = len(bound_idxs)
                        perm = torch.randperm(num_pixel)
                        K = min(num_pixel, 10)  # self.pixel_update_freq
                        feat = this_feat[:, bound_idxs[perm[:K]]]
                        feat = torch.transpose(feat, 0, 1)
                        feat_bound[lb].append(F.normalize(feat, p=2, dim=1))
                    elif self.b_update == "prob":
                        assert p2mask is not None
                        # 仅考虑MT同预测，且prob够大的代表边界点
                        bound_idxs = bound_idxs[this_p2mask[bound_idxs]]
                        num_pixel = len(bound_idxs)
                        perm = torch.randperm(num_pixel)
                        K = min(num_pixel, 10)  # self.pixel_update_freq
                        feat = this_feat[:, bound_idxs[perm[:K]]]
                        feat = torch.transpose(feat, 0, 1)
                        feat_bound[lb].append(F.normalize(feat, p=2, dim=1))
                    elif self.b_update == "prob_p":
                        assert p2mask is not None
                        # 仅考虑MT同预测，且prob够大的代表边界点
                        bound_idxs = bound_idxs[this_p2mask[bound_idxs]]
                        num_pixel = len(bound_idxs)
                        perm = torch.randperm(num_pixel)
                        if num_pixel > 10 and proto is not None:
                            # 计算与proto的相似度，然后从小到达排序
                            p2b_sim = torch.matmul(proto.unsqueeze(0), F.normalize(this_feat[:, bound_idxs],p=2,dim=0)).view(-1)
                            _, perm = torch.sort(p2b_sim, descending=True)

                        K = min(num_pixel, 10)  # self.pixel_update_freq
                        feat = this_feat[:, bound_idxs[perm[:K]]]
                        feat = torch.transpose(feat, 0, 1)
                        feat_bound[lb].append(F.normalize(feat, p=2, dim=1))

        for cls in range(self.n_classes):
            if len(feat_inner[cls]) > 0:
                rep_cls = torch.stack(feat_inner[cls], dim=0)
                self.proj_memory[0][cls] = torch.cat((rep_cls, self.proj_memory[0][cls]), dim=0)
                if self.proj_memory[0][cls].shape[0] > self.memory_per_class:
                    self.proj_memory[0][cls] = self.proj_memory[0][cls][:self.memory_per_class]

            if len(feat_bound[cls]) > 0:
                if self.b_update == "mean":
                    rep_cls = torch.stack(feat_bound[cls], dim=0)
                else:
                    rep_cls = torch.cat(feat_bound[cls], dim=0)
                self.proj_memory[1][cls] = torch.cat((rep_cls, self.proj_memory[1][cls]), dim=0)
                if self.proj_memory[1][cls].shape[0] > self.memory_per_class:
                    self.proj_memory[1][cls] = self.proj_memory[1][cls][:self.memory_per_class]


def _compute_nce(temperature, i, anchor_num, anchor_feature, y_anchor, anchor_count, contrast_feature, y_contrast,
                 contrast_count, base_temperature=0.07):
    contrast_half_size = contrast_feature.shape[0] // 2

    # (anchor_num × n_view) × [(anchor_num × n_view) | (c × mem_size)]
    anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)

    # (anchor_num × 1) × [(anchor_num × n_view) | (class_num × mem_size)]
    mask = torch.eq(y_anchor, y_contrast.T).float().cuda()
    mask = mask.repeat(anchor_count, contrast_count)
    neg_mask = 1 - mask

    # 同类另一个视图为弱正类
    if i == 0:
        mask[:, contrast_half_size:] = 0.0  # 0.0
    else:
        mask[:, :contrast_half_size] = 0.0

    anchor_dot_contrast = torch.div(anchor_dot_contrast, temperature)  # self.temperature
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # # TODO: 由于是基于bank的，所以没有自相似
    # # 去自身的相似
    # logits_mask = torch.ones_like(mask). \
    #     scatter_(1, torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(), 0)
    # mask = mask * logits_mask

    neg_logits = torch.exp(logits) * neg_mask
    neg_logits = neg_logits.sum(1, keepdim=True)  # (anchor_num × n_view) × 1
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits + neg_logits)  # -l_{ij}

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-5)  # 正类lij的平均
    loss = - (temperature / base_temperature) * mean_log_prob_pos  # self.temperature
    loss = loss.mean()

    return loss


def _compute_bcl(temperature, i, anchor_num, anchor_feature, y_anchor, anchor_count, contrast_feature, y_contrast,
                 contrast_count):
    with torch.no_grad():
        # 计算边界pixel的NCE loss
        loss_bnce = _compute_nce(temperature[0], i, anchor_num, anchor_feature, y_anchor, anchor_count,
                                 contrast_feature, y_contrast, contrast_count)  # 0.1

    soft_plus = torch.nn.Softplus()  # f(x)=log(1+exp(x))
    # 计算相似性
    # (anchor_num × n_view) × [(anchor_num × n_view) | (c × mem_size)]
    anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)
    anchor_dot_contrast = torch.div(anchor_dot_contrast, temperature[1])  # self.temperature
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    # positive 和negative累加
    # (anchor_num × 1) × [(anchor_num × n_view) | (class_num × mem_size)]
    mask = torch.eq(y_anchor, y_contrast.T).float().cuda()
    mask = mask.repeat(anchor_count, contrast_count)
    neg_mask = 1 - mask

    # 同类另一个视图为弱正类
    contrast_half_size = contrast_feature.shape[0] // 2
    if i == 0:
        mask[:, contrast_half_size:] = 0.0  # 0.0
    else:
        mask[:, :contrast_half_size] = 0.0
    # 得到loss
    sum_pred_pos = (-(logits * mask) - (1 - mask) * 1e7)
    sum_pred_neg = ((logits * neg_mask) - (1 - neg_mask) * 1e7)
    # min(s+) > max(s-) + 0.01
    loss = soft_plus(torch.logsumexp(sum_pred_pos, dim=1) + torch.logsumexp(sum_pred_neg, dim=1))  # + 0
    # 归一化到nce loss的值
    weight_balance = loss_bnce.detach().item() / loss.mean().item()
    loss *= weight_balance
    loss = loss.mean()

    return loss


def compute_spcl_loss(
        rep,
        memo,
        label,
        bound,
        prob,  # b,h,w
        temperature=[0.5, 0.07],  # 0.07
        k_low_thresh=256,  # anchor number
        strong_threshold=0.8,
):
    # proj_memory = torch.stack(memo.proj_memory[0])
    X_contrast, y_contrast = memo.concat_memory()
    contrast_feature = torch.cat(torch.unbind(X_contrast, dim=1), dim=0)  # 2N, f
    y_contrast = torch.cat(torch.unbind(y_contrast, dim=1), dim=0).view(-1, 1)  # 2N,1
    contrast_count = 1

    # Scan classes in current batches
    batch_cls = torch.unique(label)
    cls_filt = (batch_cls != 255)
    batch_cls = batch_cls[cls_filt]

    # fts = fts.permute(0, 2, 3, 1)  # B H W C
    rep = rep.permute(0, 2, 3, 1)  # B H W C

    # Begin loss calculation
    loss = [torch.tensor(0.0), torch.tensor(0.0)]
    cls_cnt = [0, 0]
    for i, cls in enumerate(batch_cls):
        cls_map = label == cls
        cls_fts_cnt = cls_map.sum()  # check number of featurs for current class

        if cls_fts_cnt == 0:  # if class fts are not enough, skip current class
            continue

        # 获取边内特征表示集
        bound_cls = bound[cls_map]  # N
        proj_cls = rep[cls_map]  # N 256
        proj_cls = F.normalize(proj_cls, p=2, dim=1)
        prob_cls = prob[cls_map]  # N
        mask_hard = prob_cls < strong_threshold  # self.strong_threshold = 0.8
        proj_cls_i = proj_cls[(bound_cls == 0.0) * mask_hard]
        proj_cls_b = proj_cls[(bound_cls == 1.0) * mask_hard]

        # # Select low density anchors
        # _, idx = torch.topk(s_to_b_density, k=low_thresh, dim=0, largest=False)
        # proj_anchor = proj_cls[idx].clone().cuda()

        # Random Sampling hard anchor representations and compute contrastive loss
        low_thresh = len(proj_cls_i) if len(proj_cls_i) < k_low_thresh else k_low_thresh  # Select all of them
        if low_thresh > 0:
            # sample_idx = torch.randint(len(proj_cls_i), size=(low_thresh,))
            sample_idx = np.random.choice(proj_cls_i.size(0), low_thresh, False)
            proj_anchor_i = proj_cls_i[sample_idx]
            anchor_num_inner = low_thresh
            y_inner = torch.tensor([cls] * anchor_num_inner).view(-1, 1).cuda()
            anchor_count_inner = 1

            loss_i = _compute_nce(temperature[0], 0, anchor_num_inner, proj_anchor_i, y_inner, anchor_count_inner,
                                  contrast_feature, y_contrast, contrast_count)
            loss[0] = loss[0] + loss_i
            cls_cnt[0] += 1

        low_thresh = len(proj_cls_b) if len(proj_cls_b) < k_low_thresh else k_low_thresh  # Select all of them
        if low_thresh > 0:
            # sample_idx = torch.randint(len(proj_cls_b), size=(low_thresh,))
            sample_idx = np.random.choice(proj_cls_b.size(0), low_thresh, False)
            proj_anchor_b = proj_cls_b[sample_idx]
            anchor_num_bound = low_thresh
            y_bound = torch.tensor([cls] * anchor_num_bound).view(-1, 1).cuda()
            anchor_count_bound = 1

            loss_b = _compute_bcl(temperature, 1, anchor_num_bound, proj_anchor_b, y_bound, anchor_count_bound,
                                  contrast_feature, y_contrast, contrast_count)
            loss[1] = loss[1] + loss_b
            cls_cnt[1] += 1

    loss[0] = loss[0] if cls_cnt[0] == 0 else loss[0] / cls_cnt[0]
    loss[1] = loss[1] if cls_cnt[1] == 0 else loss[1] / cls_cnt[1]

    return loss


