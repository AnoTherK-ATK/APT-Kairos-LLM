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

# Import các module của bạn
from config import *
from kairos_utils import *
import attack_investigation
from explainer import TemporalGNNExplainer

# --- CẤU HÌNH ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Giả định artifacts_dir, models_dir... đã được define trong config
HISTORY_FILES = [f"{artifact_dir}graph_4_{day}_history_list" for day in [6, 11, 12, 13]]
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


# [FIX] Hàm thống nhất để lấy Hash và Label từ Node ID
# Giúp đảm bảo Explainer và Louvain đều nhìn thấy cùng một Node
def get_node_data(node_id, nodeid2msg):
    label = get_label_from_db(node_id, nodeid2msg)
    # Lưu ý: hashgen phải được import từ kairos_utils
    node_hash = str(hashgen(label))
    return node_hash, label


def add_node_with_label(graph, node_hash, label):
    # Chỉ thêm nếu chưa có để tránh ghi đè thuộc tính không mong muốn
    if node_hash not in graph:
        graph.add_node(node_hash, label=label)


def apply_visual_style(graph):
    graph.graph['rankdir'] = 'LR'
    graph.graph['splines'] = 'true'
    graph.graph['nodesep'] = '0.5'
    graph.graph['ranksep'] = '1.5'
    graph.graph['overlap'] = 'false'

    for node, data in graph.nodes(data=True):
        label = data.get('label', '')
        data['color'] = 'blue'
        data['fontcolor'] = 'black'
        data['fontsize'] = '10'
        data['style'] = ''

        if 'netflow' in label or (':' in label and any(c.isdigit() for c in label)):
            data['shape'] = 'diamond'
        elif 'subject' in label or any(proc in label for proc in ['imapd', 'sh', 'python', 'nginx', 'vim']):
            data['shape'] = 'box'
        else:
            data['shape'] = 'ellipse'

    for u, v, data in graph.edges(data=True):
        data['color'] = 'blue'
        data['fontcolor'] = 'black'
        data['fontsize'] = '8'
        data['penwidth'] = '1.0'
        data['arrowsize'] = '0.8'


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
        # print(f"   Advancing stream to {target_timestamp}...")
        self.memory.eval()
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
    # [FIX] Sửa logic kiểm tra file history
    if not HISTORY_FILES:
        print("History file list is empty.")
        return

    print("Initializing Database connection...")
    cur, _ = init_database_connection()
    nodeid2msg = gen_nodeid2msg(cur)

    print(f"Loading graph data (CPU)...")
    full_data = torch.load(TEST_DATA_PATH, weights_only=False)

    memory, gnn, link_pred, neighbor_loader = load_kairos_model()
    replayer = StreamReplayer(full_data, memory, neighbor_loader, DEVICE)

    # Graphs
    critical_path = nx.DiGraph()
    louvain_input_graph = nx.DiGraph()
    summary_graph = nx.DiGraph()

    for HISTORY_FILE in HISTORY_FILES:
        if not os.path.exists(HISTORY_FILE):
            print(f"Skipping missing file: {HISTORY_FILE}")
            continue

        history_list = torch.load(HISTORY_FILE, weights_only=False)
        best_queue = max(history_list, key=lambda q: sum(tw['loss'] for tw in q))

        explainer = TemporalGNNExplainer(
            model={'gnn': gnn, 'link_pred': link_pred},
            criterion=torch.nn.CrossEntropyLoss(),
            epochs=30, lr=0.05, device=DEVICE
        )

        sorted_windows = sorted(best_queue, key=lambda x: x['name'])
        ANOMALOUS_GRAPH_DATE = f"{artifact_dir}graph_4_{HISTORY_FILE.split('_')[2]}"

        for window in sorted_windows:
            print(f"\n>>> Processing Window: {window['name']}")
            log_path = f"{ANOMALOUS_GRAPH_DATE}/{window['name']}"
            if not os.path.exists(log_path): continue

            anomalous_events = []
            with open(log_path, 'r') as f:
                for line in f: anomalous_events.append(eval(line.strip()))
            if not anomalous_events: continue

            replayer.advance_to(min([e['time'] for e in anomalous_events]))

            # --- THRESHOLD LOGIC ---
            losses = [e['loss'] for e in anomalous_events]
            mean_loss = np.mean(losses)
            std_loss = np.std(losses)
            threshold = mean_loss + 1.5 * std_loss

            # Chỉ chạy Explainer trên các sự kiện High Loss
            target_events = [e for e in anomalous_events if e['loss'] > threshold]

            if len(target_events) == 0:
                print("   [Warn] Falling back to Top-10 events.")
                target_events = sorted(anomalous_events, key=lambda x: x['loss'], reverse=True)[:10]

            print(f"   Explaining {len(target_events)} events...")

            # --- 1. BUILD CRITICAL PATH (EXPLAINER) ---
            for event in tqdm(target_events, desc="Explaining"):
                try:
                    src_ids, dst_ids, weights = explainer.explain_edge(
                        event['srcnode'], event['dstnode'], event['time'],
                        full_data, memory, neighbor_loader
                    )

                    for i in range(len(src_ids)):
                        u_id, v_id, w = src_ids[i], dst_ids[i], weights[i]

                        # [FIX] Sử dụng hàm thống nhất get_node_data
                        u_hash, u_label = get_node_data(u_id, nodeid2msg)
                        v_hash, v_label = get_node_data(v_id, nodeid2msg)

                        add_node_with_label(critical_path, u_hash, u_label)
                        add_node_with_label(critical_path, v_hash, v_label)

                        if w > 0.4:
                            critical_path.add_edge(u_hash, v_hash, weight=float(w), type='explainer')
                except Exception as e:
                    # print(f"Error explaining: {e}")
                    pass

            # --- 2. BUILD LOUVAIN INPUT (COMMUNITY DETECTION) ---
            # Sử dụng toàn bộ sự kiện có loss cao hơn loss trung bình của window
            for event in anomalous_events:
                if event['loss'] > window['loss']:
                    # [FIX QUAN TRỌNG] Thay vì parse msg text, ta dùng srcnode ID
                    # để đảm bảo Hash khớp hoàn toàn với bên Explainer
                    u_hash, u_label = get_node_data(event['srcnode'], nodeid2msg)
                    v_hash, v_label = get_node_data(event['dstnode'], nodeid2msg)

                    add_node_with_label(louvain_input_graph, u_hash, u_label)
                    add_node_with_label(louvain_input_graph, v_hash, v_label)
                    louvain_input_graph.add_edge(u_hash, v_hash)

        # --- 3. RUN LOUVAIN & UPDATE SUMMARY GRAPH ---
        if louvain_input_graph.number_of_edges() > 0:
            undirected_g = louvain_input_graph.to_undirected()
            try:
                # print(f"   Running Louvain on {louvain_input_graph.number_of_edges()} edges...")
                partition = attack_investigation.community_louvain.best_partition(undirected_g)

                # Chỉ thêm cạnh vào summary_graph nếu 2 node cùng community
                for u, v in louvain_input_graph.edges():
                    if u in partition and v in partition:
                        if partition[u] == partition[v]:
                            summary_graph.add_edge(u, v)
                        # Tùy chọn: Bạn có muốn giữ lại các cạnh nối giữa các community không?
                        # Nếu muốn giữ lại "cầu nối" tấn công, hãy bỏ điều kiện partition[u] == partition[v]
            except Exception as e:
                print(f"Louvain Error: {e}")

    # --- 4. INTERSECTION & RESULT ---
    print("\n>>> Finding Intersection (Verified Attack Path)...")

    # Debug: Kiểm tra độ chồng lặp của Node
    crit_nodes = set(critical_path.nodes())
    summ_nodes = set(summary_graph.nodes())
    overlap = crit_nodes.intersection(summ_nodes)
    print(f"Debug: Explainer Nodes: {len(crit_nodes)}, Summary Nodes: {len(summ_nodes)}")
    print(f"Debug: Overlapping Nodes: {len(overlap)}")

    # Thực hiện Intersection
    verified_graph_struct = nx.intersection(critical_path, summary_graph)

    verified_graph = nx.DiGraph()
    for u, v in verified_graph_struct.edges():
        # Lấy label từ critical_path (nơi chứa label chính xác nhất)
        u_lbl = critical_path.nodes[u].get('label', u)
        v_lbl = critical_path.nodes[v].get('label', v)
        verified_graph.add_node(u, label=u_lbl)
        verified_graph.add_node(v, label=v_lbl)
        verified_graph.add_edge(u, v)

    # --- FALLBACK NẾU MẤT CẠNH QUÁ NHIỀU ---
    # Nếu verified graph rỗng, ta có thể lấy critical_path làm kết quả chính
    # vì Explainer quan trọng hơn Louvain trong việc tìm nguyên nhân gốc rễ.
    if verified_graph.number_of_edges() == 0 and critical_path.number_of_edges() > 0:
        print("\n[WARN] Intersection resulted in 0 edges. Creating graph from Critical Path only.")
        verified_graph = critical_path.copy()

    apply_visual_style(verified_graph)
    apply_visual_style(critical_path)
    apply_visual_style(summary_graph)

    print(f"Stats:")
    print(f" - Critical Path Edges: {critical_path.number_of_edges()}")
    print(f" - Summary Graph Edges: {summary_graph.number_of_edges()}")
    print(f" - VERIFIED ATTACK EDGES: {verified_graph.number_of_edges()}")

    if verified_graph.number_of_edges() > 0:
        output_dot = f"{artifact_dir}verified_attack_path.dot"
        output_png = f"{artifact_dir}verified_attack_path.png"
        nx.drawing.nx_pydot.write_dot(verified_graph, output_dot)
        nx.drawing.nx_pydot.write_dot(summary_graph, f"{artifact_dir}summary_graph.dot")
        nx.drawing.nx_pydot.write_dot(critical_path, f"{artifact_dir}critical_path.dot")
        try:
            os.system(f"dot -Tpng {output_dot} -o {output_png}")
            print(f"SUCCESS: Result saved to {output_png}")
        except:
            print("Saved dot file (Graphviz not found/failed).")
    else:
        print("No graph generated.")


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()