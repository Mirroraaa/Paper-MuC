import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_


class Model(nn.Module):
    def __init__(self, num_entities, num_relations, dimension, parts, num_core, alpha):
        super(Model, self).__init__()
        self.dimension = dimension
        self.parts = parts
        self.num_cores = num_core
        self.p = 2
        self.q = alpha / 2
        bound = 0.01

        self.W_list = nn.ParameterList([
            nn.Parameter(torch.empty(parts, parts, parts)) for _ in range(self.num_cores)
        ])
        for W in self.W_list:
            nn.init.uniform_(W, -bound, bound)

        self.entity = nn.Embedding(num_entities, dimension)
        self.relation = nn.Embedding(num_relations, dimension)
        xavier_normal_(self.entity.weight.data)
        xavier_normal_(self.relation.weight.data)

        self.attention_layer = nn.Sequential(
            nn.Linear(2 * dimension, dimension),
            nn.ReLU(),

            nn.Linear(dimension, self.num_cores)
        )

    def geometric_consistency_regularization(self):
        if self.num_cores <= 1:
            return torch.tensor(0.0, device=self.W_list[0].device)

        # Compute the "centroid" of each core tensor
        centers = []
        for W in self.W_list:
            coords = torch.arange(self.parts, dtype=torch.float32, device=W.device)
            W_abs = torch.abs(W)

            center_x = torch.sum(W_abs * coords.view(-1, 1, 1)) / torch.sum(W_abs)
            center_y = torch.sum(W_abs * coords.view(1, -1, 1)) / torch.sum(W_abs)
            center_z = torch.sum(W_abs * coords.view(1, 1, -1)) / torch.sum(W_abs)

            centers.append(torch.stack([center_x, center_y, center_z]))

        centers = torch.stack(centers)  # [num_cores, 3]

        # Compute distances between centroids to encourage uniform distribution
        dist_matrix = torch.cdist(centers, centers, p=2)

        # Remove diagonal entries and compute minimum distances (to avoid collapse)
        mask = torch.eye(self.num_cores, device=centers.device)
        masked_dist = dist_matrix + mask * 1e6
        min_distances = torch.min(masked_dist, dim=1)[0]

        # Encourage minimum distances to be sufficiently large
        consistency_loss = -torch.mean(torch.log(min_distances + 1e-6))

        return consistency_loss

    def forward(self, heads, relations, tails):

        h = self.entity(heads)
        r = self.relation(relations)
        t = self.entity(tails)

        hr_input = torch.cat([h, r], dim=1)
        attention_weights = F.softmax(self.attention_layer(hr_input), dim=1)

        h = h.view(-1, self.dimension // self.parts, self.parts)
        r = r.view(-1, self.dimension // self.parts, self.parts)
        t = t.view(-1, self.dimension // self.parts, self.parts)

        h_norm = ((torch.abs(h) ** self.p).sum(2) ** self.q).sum(1)
        r_norm = ((torch.abs(r) ** self.p).sum(2) ** self.q).sum(1)
        t_norm = ((torch.abs(t) ** self.p).sum(2) ** self.q).sum(1)

        x1_list, x2_list, x3_list = [], [], []
        for W in self.W_list:
            temp1 = torch.matmul(h, W.view(self.parts, -1))
            temp2 = torch.matmul(r, W.permute(1, 2, 0).contiguous().view(self.parts, -1))
            temp3 = torch.matmul(t, W.permute(2, 0, 1).contiguous().view(self.parts, -1))
            x1_list.append(temp1)
            x2_list.append(temp2)
            x3_list.append(temp3)

        attn_w = attention_weights.unsqueeze(-1).unsqueeze(-1)

        x1_list2, x2_list2, x3_list2 = [], [], []
        for i in range(self.num_cores):
            temp1 = torch.matmul(r.unsqueeze(-2),
                                 x1_list[i].view(-1, self.dimension // self.parts, self.parts, self.parts)).squeeze(-2)
            temp2 = torch.matmul(t.unsqueeze(-2),x2_list[i].view(-1, self.dimension // self.parts, self.parts, self.parts)).squeeze(-2)
            temp3 = torch.matmul(h.unsqueeze(-2),x3_list[i].view(-1, self.dimension // self.parts, self.parts, self.parts)).squeeze(-2)
            x1_list2.append(temp1)
            x2_list2.append(temp2)
            x3_list2.append(temp3)
        # Combine core tensor branches using attention
        x1_stack = torch.stack(x1_list2, dim=1)
        x2_stack = torch.stack(x2_list2, dim=1)
        x3_stack = torch.stack(x3_list2, dim=1)

        x1 = (attn_w * x1_stack).sum(dim=1)
        x2 = (attn_w * x2_stack).sum(dim=1)
        x3 = (attn_w * x3_stack).sum(dim=1)

        whr_norm = ((torch.abs(x1) ** self.p).sum(2) ** self.q).sum(1)
        wrt_norm = ((torch.abs(x2) ** self.p).sum(2) ** self.q).sum(1)
        wth_norm = ((torch.abs(x3) ** self.p).sum(2) ** self.q).sum(1)

        x1 = x1.view(-1, self.dimension)
        scores = torch.matmul(x1, self.entity.weight.t())

        factor1 = torch.mean(h_norm) + torch.mean(r_norm) + torch.mean(t_norm)
        factor2 = torch.mean(whr_norm) + torch.mean(wrt_norm) + torch.mean(wth_norm)
        factor3 = self.geometric_consistency_regularization()

        return scores, factor1, factor2, factor3, attention_weights


