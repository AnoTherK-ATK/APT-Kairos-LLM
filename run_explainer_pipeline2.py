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
import argparse

from config import *
from kairos_utils import *
import attack_investigation
from explainer2 import TemporalGNNExplainer


# --- CẤU HÌNH ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HISTORY_FILE = f"{artifact_dir}graph_4_6_history_list"
MODEL_PATH = f"{models_dir}models.pt"
TEST_DATA_PATH = f"{graphs_dir}graph_4_6.TemporalData.simple"


# --- HELPER FUNCTIONS ---
def get_clean_label(msg_str):
    """
    Chuyển đổi chuỗi log thô thành nhãn sạch.
    Ví dụ: "{'file': '/usr/bin/python'}" -> "/usr/bin/python"
    """
    try:
        # Parse chuỗi thành dictionary
        msg_dict = ast.literal_eval(msg_str)
        if isinstance(msg_dict, dict):
            raw_label = list(msg_dict.values())[0]
            # Áp dụng rút gọn đường dẫn
            return attack_investigation.replace_path_name(raw_label)
    except:
        pass
    return attack_investigation.replace_path_name(msg_str)


def get_label_from_db(node_id, nodeid2msg):
    """Lấy nhãn từ ID số thông qua DB map"""
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
                if processed % 2000 == 0: print(f"   ...stream at {self.last_processed_time}")
        torch.cuda.empty_cache()


def load_kairos_model():
    print(f"Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    memory, gnn, link_pred, neighbor_loader = checkpoint
    memory.eval();
    gnn.eval();
    link_pred.eval()
    memory.reset_state();
    neighbor_loader.reset_state()
    return memory, gnn, link_pred, neighbor_loader


def main(top):
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

        # --- EXPLAINER PHASE ---
        top_events = sorted(anomalous_events, key=lambda x: x['loss'], reverse=True)[:top]
        print(f"   Running GNNExplainer on Top-{len(top_events)} events...")

        for event in tqdm(top_events, desc="Explaining"):
            try:
                src_ids, dst_ids, weights = explainer.explain_edge(
                    event['srcnode'], event['dstnode'], event['time'],
                    full_data, memory, neighbor_loader
                )

                for i in range(len(src_ids)):
                    u_id, v_id, w = src_ids[i], dst_ids[i], weights[i]

                    # Explainer dùng Global ID -> get_label_from_db -> Clean Label
                    u_label = get_label_from_db(u_id, nodeid2msg)
                    v_label = get_label_from_db(v_id, nodeid2msg)

                    u_hash = str(hashgen(u_label))
                    v_hash = str(hashgen(v_label))

                    add_node_with_label(critical_path, u_hash, u_label)
                    add_node_with_label(critical_path, v_hash, v_label)

                    if w > 0.5:
                        critical_path.add_edge(u_hash, v_hash, weight=float(w), type='explainer')
            except Exception as e:
                pass

        # --- LOUVAIN INPUT PHASE (FIXED) ---
        # Logic này xây dựng đồ thị tổng quan để chạy Louvain
        for event in anomalous_events:
            if event['loss'] > window['loss']:
                # [FIX QUAN TRỌNG]: Dùng get_clean_label thay vì replace_path_name trực tiếp
                # Để đảm bảo format giống hệt bên Explainer (vd: "/usr/bin/python" thay vì "{'file': '/usr/bin/python'}")
                src_clean = get_clean_label(event['srcmsg'])
                dst_clean = get_clean_label(event['dstmsg'])

                u_hash = str(hashgen(src_clean))
                v_hash = str(hashgen(dst_clean))

                add_node_with_label(louvain_input_graph, u_hash, src_clean)
                add_node_with_label(louvain_input_graph, v_hash, dst_clean)
                louvain_input_graph.add_edge(u_hash, v_hash)

    if louvain_input_graph.number_of_edges() == 0: return

    print(f"\n>>> Running Louvain on {louvain_input_graph.number_of_edges()} edges...")
    undirected_g = louvain_input_graph.to_undirected()
    try:
        partition = attack_investigation.community_louvain.best_partition(undirected_g)
        summary_graph = nx.DiGraph()

        # Copy labels
        for node, attr in louvain_input_graph.nodes(data=True):
            if node in partition: summary_graph.add_node(node, **attr)

        for u, v in louvain_input_graph.edges():
            if u in partition and v in partition:
                if partition[u] == partition[v]:
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

    print(f"Stats:")
    print(f" - Critical Path Edges (Explainer): {critical_path.number_of_edges()}")
    print(f" - Summary Graph Edges (Louvain): {summary_graph.number_of_edges()}")
    print(f" - VERIFIED ATTACK EDGES: {verified_graph.number_of_edges()}")

    if verified_graph.number_of_edges() > 0:
        nx.drawing.nx_pydot.write_dot(verified_graph, f"{artifact_dir}verified_attack_path.dot")
        try:
            os.system(f"dot -Tpng {artifact_dir}verified_attack_path.dot -o {artifact_dir}verified_attack_path.png")
            print(f"SUCCESS: Rendered image to {artifact_dir}verified_attack_path.png")
        except:
            print("Saved dot file.")
    else:
        print("No intersection found. Check hashing logic or thresholds.")


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    parser = argparse.ArgumentParser(description="Kairos GNNExplainer")
    parser.add_argument("--top", type=int, default=10, help="Top N?")
    args = parser.parse_args()
    main(args.top)