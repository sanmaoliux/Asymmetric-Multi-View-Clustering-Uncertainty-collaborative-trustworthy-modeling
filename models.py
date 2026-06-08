import sys
import time
import torch
import numpy as np
import pandas as pd
from dataprocessing import get_multiview_data
from metrics import calculate_metrics
from torch.utils.tensorboard import SummaryWriter


# ---------------- 预训练 ----------------
def pre_train(network_model, mv_data, batch_size, epochs, optimizer, writer=None):
    t0 = time.time()
    loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)
    mse = torch.nn.MSELoss()
    history = []

    for epoch in range(epochs):
        total = 0.
        for sub_views, _ in loader:
            alpha_list, recons, features, kl_total, alpha_fused, fused_prob = network_model(sub_views)

            loss = sum(mse(sub_views[v], recons[v]) for v in range(num_views))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        avg = total / num_samples
        history.append(avg)
        if writer:
            writer.add_scalar('pretrain/loss', avg, epoch)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[PreTrain] epoch {epoch:3d}  loss={avg:.6f}")

    print(f"Pre-training finished, time = {time.time() - t0:.2f}s")
    return history


def hybrid_edl_train(model,
                     mv_data,
                     mvc_loss_fn,
                     edl_loss_fn,
                     batch_size,
                     lambda_c,
                     ib_lambda,
                     edl_lambda,
                     temperature_l,
                     normalized,
                     epoch,
                     optimizer,
                     writer=None):
    model.train()
    loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)
    mse = torch.nn.MSELoss()
    total = 0.

    for bid, (sub_views, _) in enumerate(loader):
        alpha_list, recons, features, kl_total, alpha_fused, fused_prob = model(sub_views)
        loss_terms = []

        for i in range(num_views):
            loss_terms.append(mse(sub_views[i], recons[i]))

        loss_terms.append(ib_lambda * kl_total)

        prob_list = [alpha / torch.sum(alpha, dim=1, keepdim=True) for alpha in alpha_list]
        K = alpha_fused.size(1)
        S_fused = torch.sum(alpha_fused, dim=1, keepdim=True)
        u = K / S_fused
        reliability = torch.mean(1.0 - u).item()

        feature_align_loss = 0.0
        cluster_con_loss = 0.0
        for i in range(num_views):
            for j in range(i + 1, num_views):
                cos_sim = torch.nn.functional.cosine_similarity(features[i], features[j], dim=1)
                feature_align_loss += (1.0 - cos_sim).mean()
                cluster_con_loss += mvc_loss_fn.forward_label(prob_list[i], prob_list[j], temperature_l, normalized)

        total_con_loss = feature_align_loss + cluster_con_loss
        loss_terms.append(lambda_c * reliability * total_con_loss)

        edl_loss_val = edl_loss_fn(alpha_list, alpha_fused, epoch_num=epoch, annealing_step=200)
        loss_terms.append(edl_lambda * edl_loss_val)

        loss = sum(loss_terms)

        if torch.isnan(loss) or torch.isinf(loss):
            print("[Fatal] loss NaN/Inf, abort.")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()

    avg = total / num_samples
    if epoch % 10 == 0:
        print(f"[Hybrid Train] epoch {epoch:4d}  loss={avg:.6f}  reliability={reliability:.4f}")

    return avg

# ---------------- Inference / 评估 ----------------
@torch.no_grad()
def inference(model, mv_data, batch_size):
    model.eval()
    loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)
    soft_all, labels_all = [], []
    preds_each = [[] for _ in range(num_views)]

    for sub_views, lbl in loader:
        alpha_list, _, _, _, alpha_fused, fused_prob = model(sub_views)
        soft = fused_prob
        for v in range(num_views):
            prob_v = alpha_list[v] / torch.sum(alpha_list[v], dim=1, keepdim=True)
            preds_each[v].extend(torch.argmax(prob_v, dim=1).cpu().numpy())

        soft_all.extend(soft.cpu().numpy())
        labels_all.extend(lbl)

    soft_all = np.array(soft_all)
    labels_all = np.array(labels_all)
    pred_final = np.argmax(soft_all, axis=1)
    preds_each = [np.array(p) for p in preds_each]
    return pred_final, preds_each, labels_all


def valid(model, mv_data, batch_size):
    """验证聚类效果"""
    pred_final, preds_each, labels = inference(model, mv_data, batch_size)
    num_views = len(preds_each)

    for v in range(num_views):
        acc, nmi, pur, ari = calculate_metrics(labels, preds_each[v])
        print(f"View{v + 1}  ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f}")

    acc, nmi, pur, ari = calculate_metrics(labels, pred_final)
    print(f"[Fusion] ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f}")

    return acc, nmi, pur, ari





