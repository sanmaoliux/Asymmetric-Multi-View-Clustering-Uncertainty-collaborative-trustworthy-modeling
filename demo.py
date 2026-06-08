import argparse
import warnings
import os, random, time, sys
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from layers import AD_MVC
from loss import EvidentialLoss, DeepMVCLoss
from dataprocessing import MultiviewData, get_multiview_data
from models import pre_train, hybrid_edl_train, valid
from torch.optim import AdamW

warnings.filterwarnings("ignore")

# ---------------- Cmd line ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--db', default='MSRCv1',
                    choices=['MSRCv1', 'MNIST-USPS', 'COIL20', 'scene', 'hand', 'Fashion', 'BDGP', 'NUSWIDEOBJ', 'ORL',
                             'cifar10'],
                    help='dataset name')
parser.add_argument('--gpu', default='0')
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--mse_epochs', type=int, default=200)
parser.add_argument('--con_epochs', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=35)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--ib_lambda', type=float, default=1e-3)
parser.add_argument('--warmup_epochs', type=int, default=100)
parser.add_argument('--temperature_l', type=float, default=0.5, help='temperature for contrastive loss')
parser.add_argument('--edl_lambda', type=float, default=0.05, help='weight for evidential loss')
parser.add_argument('--normalized', type=bool, default=False)
parser.add_argument('--save_model', action='store_true', help='whether to save model')
args = parser.parse_args()

# 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_model_device(model):
    """检查模型所有参数是否在同一设备上"""
    devices = set()
    for name, param in model.named_parameters():
        devices.add(param.device)

    if len(devices) > 1:
        print(f"警告: 模型参数分布在多个设备上: {devices}")
        return False
    else:
        print(f"模型参数都在同一设备上: {next(iter(devices))}")
        return True

def auto_select_anchor_by_capacity(model, mv_data, batch_size):
    model.eval()
    loader, num_views, num_samples, num_clusters = get_multiview_data(mv_data, batch_size)

    all_features = [[] for _ in range(num_views)]
    all_alphas = [[] for _ in range(num_views)]

    with torch.no_grad():
        for sub_views, _ in loader:
            alpha_list, _, features, _, _, _ = model(sub_views)
            for v in range(num_views):
                all_features[v].append(features[v].cpu())
                all_alphas[v].append(alpha_list[v].cpu())

    capacity_scores = []
    certainty_scores = []

    for v in range(num_views):
        Z = torch.cat(all_features[v], dim=0)  # [N, D]
        total_variance = torch.var(Z, dim=0).sum().item()
        capacity_scores.append(total_variance)

        alpha_v = torch.cat(all_alphas[v], dim=0)  # [N, K]
        K = num_clusters
        S_v = torch.sum(alpha_v, dim=1)  # [N]
        u_v = K / S_v  # [N]
        certainty_v = (1.0 - u_v).mean().item()
        certainty_scores.append(certainty_v)

    capacity_scores = np.array(capacity_scores)
    capacity_scores_norm = np.log1p(capacity_scores)
    if capacity_scores_norm.sum() > 0:
        capacity_scores_norm = capacity_scores_norm / capacity_scores_norm.sum()

    final_scores = []
    print("\n==> Unsupervised Anchor Selection Metrics:")
    print("    (Information Capacity * Evidential Certainty)")
    for v in range(num_views):
        cap_score = capacity_scores_norm[v]
        cert_score = certainty_scores[v]
        final_score = cap_score * cert_score
        final_scores.append(final_score)

        print(f"  View {v}: Capacity(norm)={cap_score:.4f}, Certainty={cert_score:.4f} => Final Score={final_score:.4f}")

    best_anchor = int(np.argmax(final_scores))
    return best_anchor



# ---------------- 数据集特定超参 ----------------
if __name__ == "__main__":
    if args.db == "MSRCv1":
        args.lr = 0.0001
        args.batch_size = 35
        args.seed = 42
        args.con_epochs = 500
        args.warmup_epochs = 200
        args.ib_lambda = 1e-3

        dim_high_feature = 2000
        dim_low_feature = 1024
        dims = [256, 512, 1024]
        lambda_max = 0.1

    elif args.db == "BDGP":
        args.lr = 0.0005
        args.batch_size = 250
        args.seed = 42
        args.normalized = True
        args.con_epochs = 500
        args.warmup_epochs = 100
        args.temperature_l = 0.1

        args.ib_lambda = 1e-3

        dim_high_feature = 1024
        dim_low_feature = 512
        dims = [256, 512, 1024]

        lambda_max = 0.1



    # =================== 运行实验 ===================
    set_seed(args.seed)
    mv_train = MultiviewData(args.db, device, training=True)
    mv_eval = MultiviewData(args.db, device, training=False)

    num_views = len(mv_train.data_views)
    num_samples = mv_train.labels.size
    num_clusters = np.unique(mv_train.labels).size
    input_sizes = [mv_train.data_views[v].shape[1] for v in range(num_views)]

    print(f"[Info] Dataset: {args.db}")
    print(f"[Info] views={num_views}  samples={num_samples}  clusters={num_clusters}")

    model = AD_MVC(
        num_views, input_sizes, dims,
        dim_high_feature, dim_low_feature, num_clusters,
        teacher_index=0
    ).to(device)

    check_model_device(model)

    optim = AdamW(model.parameters(), lr=args.lr)

    # ---------- (1) 预训练 ----------
    print("==> Pre-training (MSE) ...")
    t = time.time()
    pre_train(model, mv_train, args.batch_size, args.mse_epochs, optim)

    # ---------- (2) 自动化计算并选择确定性锚点 ----------
    print("==> Automatically computing Deterministic Anchor View...")

    new_teacher = auto_select_anchor_by_capacity(model, mv_eval, args.batch_size)

    print(f"\n[Decision] Selected View {new_teacher} as the Deterministic Anchor.")
    model.set_teacher(new_teacher)


    mvc_loss_fn = DeepMVCLoss(args.batch_size, num_clusters, lambda_=lambda_max).to(device)
    edl_loss_fn = EvidentialLoss(num_clusters).to(device)

    print("==> Hybrid Engine training ...")

    for epoch in range(args.con_epochs):
        ratio = min(1.0, epoch / args.warmup_epochs)
        ib_cur = args.ib_lambda * ratio
        lam_c_cur = lambda_max * ratio

        loss = hybrid_edl_train(
            model=model,
            mv_data=mv_train,
            mvc_loss_fn=mvc_loss_fn,
            edl_loss_fn=edl_loss_fn,
            batch_size=args.batch_size,
            lambda_c=lam_c_cur,
            ib_lambda=ib_cur,
            edl_lambda=args.edl_lambda,
            temperature_l=args.temperature_l,
            normalized=args.normalized,
            epoch=epoch,
            optimizer=optim
        )


        if epoch % 100 == 99 or epoch == args.con_epochs - 1:
            print(f"==> Epoch {epoch + 1}: Intermediate evaluation...")
            _ = valid(model, mv_eval, args.batch_size)

    print("==> Training finished, final validating ...")
    acc, nmi, pur, ari = valid(model, mv_eval, args.batch_size)

    with open(f'result_{args.db}.txt', 'a+') as f:
        f.write(
            '{} \t {} \t {} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f} \t  {:.6f} \t {:.4f} \n'.format(
                args.seed, args.batch_size, args.lr, lambda_max,
                acc, nmi, pur, ari, (time.time() - t)
            ))


    if args.save_model:
        save_dict = {
            'model_state_dict': model.state_dict(),
            'performance': {'acc': acc, 'nmi': nmi, 'pur': pur, 'ari': ari},
            'fusion_mode': 'Dempster-Shafer_Evidence'
        }
        torch.save(save_dict, f'model_{args.db}_EDL.pth')

    print("\n=== Experiment Summary ===")
    print(f"Dataset: {args.db}")
    print(f"Fusion mode: Dempster-Shafer Evidence (Parameter-Free)")
    print(f"Final performance: ACC={acc:.4f}, NMI={nmi:.4f}, PUR={pur:.4f}, ARI={ari:.4f}")