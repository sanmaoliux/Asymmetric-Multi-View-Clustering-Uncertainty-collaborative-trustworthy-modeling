import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepMVCLoss(nn.Module):
    def __init__(self, num_samples, num_clusters, lambda_):
        super(DeepMVCLoss, self).__init__()
        self.num_samples = num_samples
        self.num_clusters = num_clusters
        self.lambda_ = lambda_
        # self.beta = beta

        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask.fill_diagonal_(0)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        return mask.bool()


    def forward_label(self, q_i, q_j, temperature_l, normalized=False):
        q_i = self.target_distribution(q_i)
        q_j = self.target_distribution(q_j)

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.num_clusters
        q = torch.cat((q_i, q_j), dim=0)

        if normalized:
            sim = (self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / temperature_l).to(q.device)
        else:
            sim = (torch.matmul(q, q.T) / temperature_l).to(q.device)

        sim_i_j = torch.diag(sim, self.num_clusters)
        sim_j_i = torch.diag(sim, -self.num_clusters)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N).to(q.device)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss

    def target_distribution(self, q):
        weight = (q ** 2.0) / torch.sum(q, 0)
        return (weight.t() / torch.sum(weight, 1)).t()


class EvidentialLoss(nn.Module):

    def __init__(self, num_clusters):
        super(EvidentialLoss, self).__init__()
        self.num_clusters = num_clusters

    def kl_divergence(self, alpha, device):
        beta = torch.ones([1, self.num_clusters], dtype=torch.float32, device=device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)

        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)

        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl

    def forward(self, alpha_list, alpha_fused, epoch_num, annealing_step=50):
        device = alpha_fused.device

        fused_prob = alpha_fused / torch.sum(alpha_fused, dim=1, keepdim=True)

        weight = (fused_prob ** 2.0) / torch.sum(fused_prob, 0)
        Y = (weight.t() / torch.sum(weight, 1)).t().detach()

        total_loss = 0.0
        annealing_coef = min(1.0, epoch_num / annealing_step)

        for alpha in alpha_list:
            S = torch.sum(alpha, dim=1, keepdim=True)

            ece_loss = torch.sum(Y * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

            alpha_tilde = Y + (1 - Y) * alpha
            kl_loss = self.kl_divergence(alpha_tilde, device)

            view_loss = (ece_loss + annealing_coef * kl_loss).mean()
            total_loss += view_loss

        return total_loss