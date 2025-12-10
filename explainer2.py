import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


class TemporalGNNExplainer(nn.Module):
    def __init__(self, model, criterion, epochs=50, lr=0.01, device='cpu'):
        super(TemporalGNNExplainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.epochs = epochs
        self.lr = lr
        self.device = device

    def explain_edge(self, src_idx, dst_idx, t_idx, data, memory, neighbor_loader):
        self.model['gnn'].eval()
        self.model['link_pred'].eval()
        memory.eval()

        # 1. Lấy Computation Graph
        src_tensor = torch.tensor([src_idx], dtype=torch.long, device=self.device)
        dst_tensor = torch.tensor([dst_idx], dtype=torch.long, device=self.device)

        # n_id chứa GLOBAL ID của các node trong subgraph
        n_id = torch.cat([src_tensor, dst_tensor]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)

        e_id_cpu = e_id.cpu()
        subgraph_t = data.t[e_id_cpu].to(self.device)
        subgraph_msg = data.msg[e_id_cpu].to(self.device)

        # 2. Init Mask
        num_edges = edge_index.size(1)
        edge_mask = nn.Parameter(torch.rand(num_edges, device=self.device))
        optimizer = torch.optim.Adam([edge_mask], lr=self.lr)

        # 3. Memory Snapshot
        with torch.no_grad():
            z_original, last_update = memory(n_id)

        assoc = torch.empty(memory.num_nodes, dtype=torch.long, device=self.device)
        assoc[n_id] = torch.arange(n_id.size(0), device=self.device)
        target_src_local = assoc[src_idx]
        target_dst_local = assoc[dst_idx]

        # Target Label Logic (Giữ nguyên)
        target_edge_mask = (data.src[e_id_cpu] == src_idx) & \
                           (data.dst[e_id_cpu] == dst_idx) & \
                           (data.t[e_id_cpu] == t_idx)
        target_indices = torch.nonzero(target_edge_mask, as_tuple=True)[0]
        if len(target_indices) > 0:
            target_msg = subgraph_msg[target_indices[0]]
        else:
            target_msg = subgraph_msg[0]

        node_feat_dim = 16
        edge_feat = target_msg[node_feat_dim:-node_feat_dim]
        target_label = torch.argmax(edge_feat).unsqueeze(0)

        # 4. Optimization Loop (Giảm số vòng lặp hiển thị để đỡ spam)
        # Giảm learning rate một chút để ổn định hơn
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            mask_sigmoid = torch.sigmoid(edge_mask)

            z = self.model['gnn'](z_original, last_update, edge_index, subgraph_t, subgraph_msg,
                                  edge_weight=mask_sigmoid)
            pos_out = self.model['link_pred'](z[target_src_local], z[target_dst_local])

            loss_pred = self.criterion(pos_out.unsqueeze(0), target_label)
            loss_mask = torch.sum(mask_sigmoid) * 0.005
            loss_entropy = -torch.sum(mask_sigmoid * torch.log(mask_sigmoid + 1e-8) +
                                      (1 - mask_sigmoid) * torch.log(1 - mask_sigmoid + 1e-8)) * 0.01

            total_loss = loss_pred + loss_mask + loss_entropy
            total_loss.backward(retain_graph=True)
            optimizer.step()

        # 5. Trích xuất Subgraph (QUAN TRỌNG)
        mask_final = torch.sigmoid(edge_mask).detach().cpu().numpy()

        # Hạ ngưỡng lọc xuống 0.5 để lấy nhiều ngữ cảnh hơn
        important_indices = np.where(mask_final > 0.5)[0]

        # Nếu ít quá thì lấy Top 10 cạnh quan trọng nhất
        if len(important_indices) < 3:
            important_indices = np.argsort(mask_final)[-10:]

        # Lấy Local Index của cạnh
        imp_local_edges = edge_index[:, important_indices]  # [2, num_imp]

        # [NEW] Map ngược từ Local Index -> Global ID để trả về
        # n_id đang chứa global ID. imp_local_edges chứa index trỏ vào n_id
        src_global = n_id[imp_local_edges[0]].cpu().numpy()
        dst_global = n_id[imp_local_edges[1]].cpu().numpy()

        # Trả về danh sách các cặp (u, v) global ID và trọng số mask
        return src_global, dst_global, mask_final[important_indices]