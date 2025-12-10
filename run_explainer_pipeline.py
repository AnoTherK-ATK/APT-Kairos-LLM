import torch
import numpy as np
import networkx as nx
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.loader import TemporalDataLoader
import gc
import ast

from config import *
from kairos_utils import *
import attack_investigation
from explainer import TemporalGNNExplainer

# --- CẤU HÌNH ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HISTORY_FILE = f"{artifact_dir}graph_4_6_history_list"
MODEL_PATH = f"{models_dir}models.pt"
TEST_DATA_PATH = f"{graphs_dir}graph_4_6.TemporalData.simple"


# --- HELPER FUNCTIONS ---
def get_clean_label(msg_str):
    try:
        msg_dict = ast.literal_eval(msg_str)
        if isinstance(msg_dict, dict):
            raw_label = list(msg_dict.values())[0]
            return attack_investigation.replace_path_name(raw_label)
    except:
        pass
    return attack_investigation.replace_path_name(msg_str)


def get_label_from_db(node_id, nodeid2msg):
    try:
        if node_id in nodeid2msg:
            info_dict = nodeid2msg[node_id]
            if isinstance(info_dict, dict):
                raw = list(info_dict.values())[0]
                return attack_investigation.replace_path_name(raw)
            return str(info_dict)
    except:
        pass
    return f"Node_{node_id}"


def add_node_with_label(graph, node_hash, label):
    graph.add_node(node_hash, label=label)


def apply_visual_style(graph):
    """
    [NEW] Hàm này biến đổi đồ thị để giống hình mẫu (Blue style)
    """
    # 1. Cấu hình chung cho Graph
    graph.graph['rankdir'] = 'LR'  # Bố cục Trái -> Phải
    graph.graph['splines'] = 'true'  # Đường nối cong mềm mại
    graph.graph['nodesep'] = '0.5'  # Khoảng cách giữa các node cùng cấp
    graph.graph['ranksep'] = '1.5'  # Khoảng cách giữa các cấp (layers)
    graph.graph['overlap'] = 'false'  # Tránh đè node

    # 2. Style cho từng Node
    for node, data in graph.nodes(data=True):
        label = data.get('label', '')

        # Mặc định chung
        data['color'] = 'blue'  # Viền xanh
        data['fontcolor'] = 'black'  # Chữ đen
        data['fontsize'] = '10'
        data['style'] = ''  # Không tô màu nền (trong suốt/trắng)

        # Logic chọn hình dáng (Shape) dựa trên nội dung Label
        # Lưu ý: Cần điều chỉnh logic string matching tùy theo dữ liệu thực tế
        if 'netflow' in label or ':' in label and any(c.isdigit() for c in label):
            # IP/Socket -> Hình thoi
            data['shape'] = 'diamond'
        elif 'subject' in label or any(proc in label for proc in ['imapd', 'sh', 'python', 'nginx', 'vim']):
            # Process -> Hình chữ nhật
            data['shape'] = 'box'
        else:
            # File -> Hình bầu dục
            data['shape'] = 'ellipse'

    # 3. Style cho từng Edge (Cạnh)
    for u, v, data in graph.edges(data=True):
        data['color'] = 'blue'  # Đường nối màu xanh
        data['fontcolor'] = 'black'  # Nhãn sự kiện màu đen
        data['fontsize'] = '8'  # Font chữ nhỏ cho cạnh
        data['penwidth'] = '1.0'  # Độ dày mảnh
        data['arrowsize'] = '0.8'  # Mũi tên vừa phải

# --- STREAM REPLAYER ---
class StreamReplayer:
    def __init__(self, data, memory, neighbor_loader, device):
        self.loader = TemporalDataLoader(data, batch_size=BATCH)
        self.iterator = iter(self.loader)
        self.memory = memory
        self.neighbor_loader = neighbor_loader
        self.device = device
        self.last_processed_time = 0

    def advance_to(self, target_timestamp):
        if self.last_processed_time >= target_timestamp: return
        print(f"   Advancing stream to {target_timestamp}...")
        self.memory.eval()
        processed = 0
        with torch.no_grad():
            while self.last_processed_time < target_timestamp:
                try:
                    batch = next(self.iterator)
                except StopIteration:
                    self.last_processed_time = float('inf')
                    break
                src, dst, t, msg = batch.src.to(self.device), batch.dst.to(self.device), batch.t.to(
                    self.device), batch.msg.to(self.device)
                n_id = torch.cat([src, dst]).unique()
                self.neighbor_loader(n_id)
                self.memory.update_state(src, dst, t, msg)
                self.neighbor_loader.insert(src, dst)
                self.memory.detach()
                self.last_processed_time = batch.t[-1].item()
                processed += 1
        torch.cuda.empty_cache()


def load_kairos_model():
    print(f"Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    memory, gnn, link_pred, neighbor_loader = checkpoint
    memory.eval()
    gnn.eval()
    link_pred.eval()
    memory.reset_state()
    neighbor_loader.reset_state()
    return memory, gnn, link_pred, neighbor_loader


def main():
    if not os.path.exists(HISTORY_FILE):
        print("History file not found.");
        return

    print("Initializing Database connection...")
    cur, _ = init_database_connection()
    nodeid2msg = gen_nodeid2msg(cur)

    print(f"Loading graph data (CPU)...")
    full_data = torch.load(TEST_DATA_PATH, weights_only=False)

    memory, gnn, link_pred, neighbor_loader = load_kairos_model()
    replayer = StreamReplayer(full_data, memory, neighbor_loader, DEVICE)

    history_list = torch.load(HISTORY_FILE, weights_only=False)
    best_queue = max(history_list, key=lambda q: sum(tw['loss'] for tw in q))

    explainer = TemporalGNNExplainer(
        model={'gnn': gnn, 'link_pred': link_pred},
        criterion=torch.nn.CrossEntropyLoss(),
        epochs=30, lr=0.05, device=DEVICE
    )

    critical_path = nx.DiGraph()
    louvain_input_graph = nx.DiGraph()

    sorted_windows = sorted(best_queue, key=lambda x: x['name'])

    for window in sorted_windows:
        print(f"\n>>> Processing Window: {window['name']}")
        log_path = f"{ANOMALOUS_GRAPH_DATE}/{window['name']}"
        if not os.path.exists(log_path): continue

        anomalous_events = []
        with open(log_path, 'r') as f:
            for line in f: anomalous_events.append(eval(line.strip()))
        if not anomalous_events: continue

        replayer.advance_to(min([e['time'] for e in anomalous_events]))

        # [UPDATE: THRESHOLD LOGIC]
        # Tính thống kê Loss trong Time Window hiện tại
        losses = [e['loss'] for e in anomalous_events]
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)

        # Thiết lập ngưỡng (Mean + 3*Sigma: Bắt sự kiện cực kỳ bất thường)
        # Bạn có thể giảm số 3 xuống 2 hoặc 1.5 nếu muốn bắt nhiều hơn
        threshold = mean_loss + 1.5 * std_loss

        # Lọc các sự kiện vượt ngưỡng
        target_events = [e for e in anomalous_events if e['loss'] > threshold]

        print(f"   [Stats] Mean Loss: {mean_loss:.4f} | Std: {std_loss:.4f} | Threshold (3-sigma): {threshold:.4f}")
        print(
            f"   [Filter] Found {len(target_events)} critical events (out of {len(anomalous_events)}) exceeding threshold.")

        # Fallback: Nếu ngưỡng quá cao không bắt được gì, lấy Top 10 để đảm bảo có kết quả
        if len(target_events) == 0:
            print("   [Warn] No events exceed threshold. Falling back to Top-10.")
            target_events = sorted(anomalous_events, key=lambda x: x['loss'], reverse=True)[:10]

        for event in tqdm(target_events, desc="Explaining"):
            try:
                src_ids, dst_ids, weights = explainer.explain_edge(
                    event['srcnode'], event['dstnode'], event['time'],
                    full_data, memory, neighbor_loader
                )

                for i in range(len(src_ids)):
                    u_id, v_id, w = src_ids[i], dst_ids[i], weights[i]
                    u_label = get_label_from_db(u_id, nodeid2msg)
                    v_label = get_label_from_db(v_id, nodeid2msg)

                    u_hash, v_hash = str(hashgen(u_label)), str(hashgen(v_label))
                    add_node_with_label(critical_path, u_hash, u_label)
                    add_node_with_label(critical_path, v_hash, v_label)

                    if w > 0.1:
                        critical_path.add_edge(u_hash, v_hash, weight=float(w), type='explainer')
            except Exception as e:
                pass

        # --- LOUVAIN INPUT (Dùng get_clean_label) ---
        for event in anomalous_events:
            # Vẫn dùng logic cũ cho Louvain (Loss > Avg của Window) để có cái nhìn tổng quan
            if event['loss'] > window['loss']:
                src_clean = get_clean_label(event['srcmsg'])
                dst_clean = get_clean_label(event['dstmsg'])
                u_hash, v_hash = str(hashgen(src_clean)), str(hashgen(dst_clean))

                add_node_with_label(louvain_input_graph, u_hash, src_clean)
                add_node_with_label(louvain_input_graph, v_hash, dst_clean)
                louvain_input_graph.add_edge(u_hash, v_hash)

    if louvain_input_graph.number_of_edges() == 0: return

    print(f"\n>>> Running Louvain on {louvain_input_graph.number_of_edges()} edges...")
    undirected_g = louvain_input_graph.to_undirected()
    try:
        partition = attack_investigation.community_louvain.best_partition(undirected_g)
        summary_graph = nx.DiGraph()
        for node, attr in louvain_input_graph.nodes(data=True):
            if node in partition: summary_graph.add_node(node, **attr)
        for u, v in louvain_input_graph.edges():
            if u in partition and v in partition and partition[u] == partition[v]:
                summary_graph.add_edge(u, v)
    except:
        summary_graph = nx.DiGraph()

    print("\n>>> Finding Intersection (Verified Attack Path)...")
    verified_graph_struct = nx.intersection(critical_path, summary_graph)

    verified_graph = nx.DiGraph()
    for u, v in verified_graph_struct.edges():
        u_lbl = critical_path.nodes[u].get('label', u)
        v_lbl = critical_path.nodes[v].get('label', v)
        verified_graph.add_node(u, label=u_lbl)
        verified_graph.add_node(v, label=v_lbl)
        verified_graph.add_edge(u, v)

    apply_visual_style(verified_graph)
    print(f"Stats:")
    print(f" - Critical Path Edges: {critical_path.number_of_edges()}")
    print(f" - Summary Graph Edges: {summary_graph.number_of_edges()}")
    print(f" - VERIFIED ATTACK EDGES: {verified_graph.number_of_edges()}")

    if verified_graph.number_of_edges() > 0:
        nx.drawing.nx_pydot.write_dot(verified_graph, f"{artifact_dir}verified_attack_path.dot")
        try:
            os.system(f"dot -Tpng {artifact_dir}verified_attack_path.dot -o {artifact_dir}verified_attack_path.png")
            print(f"SUCCESS: Result saved to {artifact_dir}verified_attack_path.png")
        except:
            print("Saved dot file.")
    else:
        print("No intersection found.")


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()