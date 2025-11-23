from kairos_utils import *
from config import *
from torch_geometric.nn import TGNMemory, TransformerConv, SAGEConv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

max_node_num = 268243  # the number of nodes in node2id table +1
min_dst_idx, max_dst_idx = 0, max_node_num
# Helper vector to map global node indices to local ones.
assoc = torch.empty(max_node_num, dtype=torch.long, device=device)



class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels, heads=8,
                                    dropout=0.0, edge_dim=edge_dim)
        self.conv2 = TransformerConv(out_channels * 8, out_channels, heads=1, concat=False,
                                     dropout=0.0, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        last_update.to(device)
        x = x.to(device)
        t = t.to(device)
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        x = F.relu(self.conv(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return x


class GraphSAGEEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GraphSAGEEmbedding, self).__init__()


        self.time_enc = time_enc


        self.conv1 = SAGEConv(in_channels, out_channels * 8, aggr="mean")
        self.conv2 = SAGEConv(out_channels * 8, out_channels, aggr="mean")

    def forward(self, x, last_update, edge_index, t, msg):

        x = x.to(device)
        edge_index = edge_index.to(device)


        x = F.relu(self.conv1(x, edge_index))

        x = self.conv2(x, edge_index)

        return x

from torch_geometric.nn import GCNConv
class GCNEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GCNEmbedding, self).__init__()
        self.time_enc = time_enc

        # GCNConv cơ bản
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)

    def forward(self, x, last_update, edge_index, t, msg):
        x = x.to(device)
        edge_index = edge_index.to(device)

        # Lớp 1 + ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Lớp 2
        x = self.conv2(x, edge_index)

        return x


from torch_geometric.nn import GATv2Conv


class GATEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GATEmbedding, self).__init__()
        self.time_enc = time_enc

        # Tính toán kích thước edge_dim
        # msg_dim là kích thước vector thông điệp (đã bao gồm time encoding nếu bạn concat)
        # Trong KAIROS gốc, edge_attr = time_encoding + msg
        edge_dim = msg_dim + time_enc.out_channels

        # GATv2 có hỗ trợ edge_dim
        self.conv1 = GATv2Conv(in_channels, out_channels, heads=2, edge_dim=edge_dim)
        # Lớp 2: input là out_channels * heads
        self.conv2 = GATv2Conv(out_channels * 2, out_channels, heads=1, edge_dim=edge_dim, concat=False)

    def forward(self, x, last_update, edge_index, t, msg):
        x = x.to(device)
        edge_index = edge_index.to(device)
        t = t.to(device)
        last_update = last_update.to(device)

        # --- Tái tạo đặc trưng cạnh (Edge Features) ---
        # Giống hệt logic trong UniMP cũ để tận dụng thông tin cạnh
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)

        # Truyền edge_attr vào GAT
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)

        return x


from torch_geometric.nn import RGCNConv
import torch


class RGCNEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc, num_relations):
        super(RGCNEmbedding, self).__init__()
        self.time_enc = time_enc
        self.num_relations = num_relations

        # RGCNConv lớp 1
        # num_relations: số lượng loại quan hệ (ví dụ: 10 loại READ, WRITE...)
        self.conv1 = RGCNConv(in_channels, out_channels, num_relations=num_relations)

        # RGCNConv lớp 2
        self.conv2 = RGCNConv(out_channels, out_channels, num_relations=num_relations)

    def forward(self, x, last_update, edge_index, t, msg):
        x = x.to(device)
        edge_index = edge_index.to(device)

        # --- BƯỚC QUAN TRỌNG: TRÍCH XUẤT LOẠI CẠNH (EDGE TYPE) ---
        # Trong KAIROS, 'msg' chứa thông tin node nguồn, node đích và loại cạnh.
        # Chúng ta cần lấy ra chỉ số loại cạnh từ vector 'msg'.
        # Giả sử cấu trúc msg là [NodeFeat | EdgeOneHot | NodeFeat]
        # Chúng ta lấy phần giữa (EdgeOneHot) và tìm vị trí số 1 (argmax)

        # Lấy kích thước feature của node từ msg (giả định msg_dim đã biết hoặc tính toán)
        # Trong test.py logic là: tensor_find(m[node_embedding_dim:-node_embedding_dim], 1)
        # Ở đây ta làm tương tự bằng torch.argmax để nhanh hơn trên GPU

        # msg shape: [num_edges, msg_dim]
        # Node embedding dim lấy từ config (node_embedding_dim)

        # Cắt lấy phần one-hot của cạnh
        edge_part = msg[:, node_embedding_dim: -node_embedding_dim]

        # Chuyển one-hot thành chỉ số (0, 1, 2...) -> Đây chính là edge_type
        edge_type = edge_part.argmax(dim=1).to(device)

        # --- ĐƯA VÀO MÔ HÌNH ---

        # Lớp 1: Truyền edge_type vào
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)

        # Lớp 2
        x = self.conv2(x, edge_index, edge_type)

        return x

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinkPredictor, self).__init__()
        self.lin_src = Linear(in_channels, in_channels * 2)
        self.lin_dst = Linear(in_channels, in_channels * 2)

        self.lin_seq = nn.Sequential(

            Linear(in_channels * 4, in_channels * 8),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 8, in_channels * 2),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 2, int(in_channels // 2)),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(int(in_channels // 2), out_channels)
        )

    def forward(self, z_src, z_dst):
        h = torch.cat([self.lin_src(z_src), self.lin_dst(z_dst)], dim=-1)
        h = self.lin_seq(h)
        return h

def cal_pos_edges_loss_multiclass(link_pred_ratio,labels):
    loss=[]
    for i in range(len(link_pred_ratio)):
        loss.append(criterion(link_pred_ratio[i].reshape(1,-1),labels[i].reshape(-1)))
    return torch.tensor(loss)
