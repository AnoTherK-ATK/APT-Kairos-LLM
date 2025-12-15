import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


class TemporalGNNExplainer(nn.Module):
    def __init__(self, model, criterion, epochs=50, lr=0.01, device='cpu'):
        super(TemporalGNNExplainer, self).__init__()
        self.model = model  # Dictionary: {'gnn': ..., 'link_pred': ...}
        self.criterion = criterion
        self.epochs = epochs
        self.lr = lr
        self.device = device

    def explain_edge(self, src_idx, dst_idx, t_idx, data, memory, neighbor_loader):
        """
        Giải thích dựa trên RECONSTRUCTION ERROR.
        Mục tiêu: Tìm subgraph sao cho Loss (RE) của nó gần với Loss bất thường gốc nhất.
        """
        self.model['gnn'].eval()
        self.model['link_pred'].eval()
        memory.eval()

        # 1. Lấy Computation Graph & Data
        src_tensor = torch.tensor([src_idx], dtype=torch.long, device=self.device)
        dst_tensor = torch.tensor([dst_idx], dtype=torch.long, device=self.device)

        n_id = torch.cat([src_tensor, dst_tensor]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)

        e_id_cpu = e_id.cpu()
        subgraph_t = data.t[e_id_cpu].to(self.device)
        subgraph_msg = data.msg[e_id_cpu].to(self.device)

        # 2. Init Mask
        num_edges = edge_index.size(1)
        # Khởi tạo mask hơi cao một chút để bắt đầu từ trạng thái "gần giống gốc"
        edge_mask = nn.Parameter(torch.randn(num_edges, device=self.device) * 0.1 + 0.5)
        optimizer = torch.optim.Adam([edge_mask], lr=self.lr)

        # 3. Snapshot Memory
        with torch.no_grad():
            z_original, last_update = memory(n_id)

        # Index Mapping
        assoc = torch.empty(memory.num_nodes, dtype=torch.long, device=self.device)
        assoc[n_id] = torch.arange(n_id.size(0), device=self.device)
        target_src_local = assoc[src_idx]
        target_dst_local = assoc[dst_idx]

        # Lấy Ground Truth Label
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

        # --- BƯỚC QUAN TRỌNG: TÍNH RECONSTRUCTION ERROR GỐC (ORIGINAL RE) ---
        # Đây là mức độ bất thường ban đầu mà ta muốn giải thích
        with torch.no_grad():
            # Forward không có mask (edge_weight=1.0)
            z_full = self.model['gnn'](
                z_original, last_update, edge_index, subgraph_t, subgraph_msg, edge_weight=None
            )
            pos_out_full = self.model['link_pred'](z_full[target_src_local], z_full[target_dst_local])
            original_re_loss = self.criterion(pos_out_full.unsqueeze(0), target_label)

        # 4. Optimization Loop (Loss-based Explanation)
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            mask_sigmoid = torch.sigmoid(edge_mask)

            # Forward với Mask
            z_masked = self.model['gnn'](
                z_original, last_update, edge_index, subgraph_t, subgraph_msg, edge_weight=mask_sigmoid
            )
            pos_out_masked = self.model['link_pred'](z_masked[target_src_local], z_masked[target_dst_local])

            # Tính RE của phiên bản đã che (Masked RE)
            masked_re_loss = self.criterion(pos_out_masked.unsqueeze(0), target_label)

            # --- HÀM MỤC TIÊU MỚI ---
            # 1. Loss Distance: Muốn Masked RE càng giống Original RE càng tốt
            # (Nghĩa là: Những cạnh giữ lại phải là nguyên nhân gây ra Loss cao đó)
            loss_dist = torch.abs(masked_re_loss - original_re_loss)

            # 2. Sparsity: Muốn mask càng nhỏ càng tốt
            loss_mask = torch.sum(mask_sigmoid) * 0.05

            # 3. Entropy: Mask nên rõ ràng (0 hoặc 1)
            loss_entropy = -torch.sum(mask_sigmoid * torch.log(mask_sigmoid + 1e-8) +
                                      (1 - mask_sigmoid) * torch.log(1 - mask_sigmoid + 1e-8)) * 0.1

            # Tổng hợp Loss
            total_loss = loss_dist + loss_mask + loss_entropy

            total_loss.backward(retain_graph=True)
            optimizer.step()

        # 5. Trích xuất Subgraph
        mask_final = torch.sigmoid(edge_mask).detach().cpu().numpy()

        # Lọc cạnh quan trọng
        important_indices = np.where(mask_final > 0.6)[0]

        if len(important_indices) < 2:
            important_indices = np.argsort(mask_final)[-5:]

        src_global = n_id[edge_index[0, important_indices]].cpu().numpy()
        dst_global = n_id[edge_index[1, important_indices]].cpu().numpy()
        weights_global = mask_final[important_indices]

        # [NEW] Trích xuất Edge Type từ message
        # Giả định cấu trúc message: [src_feat | edge_feat | dst_feat]
        # node_feat_dim = 16 (theo code của bạn)
        selected_msgs = subgraph_msg[important_indices]

        # Cắt lấy phần edge_feat ở giữa
        edge_feats = selected_msgs[:, node_feat_dim:-node_feat_dim]

        # Lấy chỉ số loại cạnh (ví dụ: 0, 1, 2...)
        edge_types_idx = torch.argmax(edge_feats, dim=1).cpu().numpy()

        # Trả về thêm edge_types_idx
        return src_global, dst_global, weights_global, edge_types_idx