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


def find_lca_for_set(graph, nodes_of_interest):
    """
    Tìm Lowest Common Ancestor (LCA) cho một tập hợp các node.
    LCA là node tổ tiên chung của tất cả các node trong tập hợp,
    và là node nằm 'thấp nhất' (xa root nhất/gần các node con nhất).
    """
    if not nodes_of_interest:
        return None

    nodes_list = list(nodes_of_interest)

    # Bước 1: Lấy node đầu tiên làm cơ sở
    if nodes_list[0] not in graph:
        return None

    # Tập hợp tổ tiên chung khởi tạo bằng tổ tiên của node đầu tiên + chính nó
    common_ancestors = nx.ancestors(graph, nodes_list[0])
    common_ancestors.add(nodes_list[0])

    # Bước 2: Giao với tổ tiên của các node còn lại
    for node in nodes_list[1:]:
        if node not in graph:
            continue
        curr_ancestors = nx.ancestors(graph, node)
        curr_ancestors.add(node)
        common_ancestors.intersection_update(curr_ancestors)

    if not common_ancestors:
        print("   [Info] No single common ancestor found for all nodes (disjoint paths).")
        return None

    # Bước 3: Tìm LCA "thấp nhất" trong số các tổ tiên chung
    # LCA là node mà KHÔNG có node con nào nằm trong tập common_ancestors
    # (Tức là nó là điểm cuối cùng của phần chung trước khi rẽ nhánh)
    lca_node = None
    max_depth = -1

    # Cách đơn giản: Chọn node trong common_ancestors có đường đi dài nhất từ một root nào đó
    # Hoặc đơn giản hơn: Kiểm tra node nào không có out-edge tới bất kỳ node nào khác trong common_ancestors

    candidates = []
    for anc in common_ancestors:
        is_lowest = True
        # Kiểm tra xem anc có trỏ tới node nào khác trong common_ancestors không
        # Nếu có, nghĩa là anc nằm "trên", chưa phải lowest
        for neighbor in graph.successors(anc):
            if neighbor in common_ancestors:
                is_lowest = False
                break
        if is_lowest:
            candidates.append(anc)

    if candidates:
        # Nếu có nhiều ứng viên (do đồ thị phức tạp), chọn node có bậc ra (out-degree) lớn nhất trong graph gốc
        # hoặc đơn giản lấy cái đầu tiên.
        lca_node = candidates[0]

    return lca_node


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

            print(f"total events: {len(anomalous_events)} | threshold: {threshold}")
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

                    timestamp = ns_time_to_datetime(event['time'])

                    for i in range(len(src_ids)):
                        u_id, v_id, w = src_ids[i], dst_ids[i], weights[i]

                        # [FIX] Sử dụng hàm thống nhất get_node_data
                        u_hash, u_label = get_node_data(u_id, nodeid2msg)
                        v_hash, v_label = get_node_data(v_id, nodeid2msg)

                        add_node_with_label(critical_path, u_hash, u_label)
                        add_node_with_label(critical_path, v_hash, v_label)

                        if w > 0.1:
                            critical_path.add_edge(u_hash, v_hash, weight=float(w), type='explainer', label=timestamp)
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
                # --- 3. RUN LOUVAIN & UPDATE SUMMARY GRAPH ---
            if louvain_input_graph.number_of_edges() > 0:
                undirected_g = louvain_input_graph.to_undirected()
                try:
                    # print(f"   Running Louvain on {louvain_input_graph.number_of_edges()} edges...")
                    partition = attack_investigation.community_louvain.best_partition(undirected_g)

                    # Chỉ thêm cạnh vào summary_graph nếu 2 node cùng community
                    for u, v in louvain_input_graph.edges():
                        if u in partition and v in partition:
                            if partition[u] == partition[v]:  # Cùng cộng đồng -> Giữ lại
                                summary_graph.add_edge(u, v)

                                # [FIX] Copy label từ louvain_input_graph sang summary_graph
                                # Vì add_edge không tự động copy thuộc tính node
                                if 'label' not in summary_graph.nodes[u]:
                                    summary_graph.nodes[u]['label'] = louvain_input_graph.nodes[u].get('label', u)

                                if 'label' not in summary_graph.nodes[v]:
                                    summary_graph.nodes[v]['label'] = louvain_input_graph.nodes[v].get('label', v)

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

        edge_data = critical_path.get_edge_data(u, v)
        edge_label = edge_data.get('label', '') if edge_data else ''

        verified_graph.add_node(u, label=u_lbl)
        verified_graph.add_node(v, label=v_lbl)
        verified_graph.add_edge(u, v, label=edge_label)

    # --- FALLBACK NẾU MẤT CẠNH QUÁ NHIỀU ---
    # Nếu verified graph rỗng, ta có thể lấy critical_path làm kết quả chính
    # vì Explainer quan trọng hơn Louvain trong việc tìm nguyên nhân gốc rễ.
    if verified_graph.number_of_edges() == 0 and critical_path.number_of_edges() > 0:
        print("\n[WARN] Intersection resulted in 0 edges. Creating graph from Critical Path only.")
        verified_graph = critical_path.copy()

    # --- 5. FIND LCA (ROOT CAUSE ANALYSIS) ---
    print("\n>>> Identifying Root Cause (LCA)...")

    # Lấy danh sách các node trong đồ thị kết quả (Verified Graph)
    target_nodes = set(verified_graph.nodes())

    # Tìm LCA dựa trên kiến thức toàn vẹn của Critical Path (Explainer)
    lca_node = find_lca_for_set(critical_path, target_nodes)

    lca_label = "Unknown"
    if lca_node:
        lca_label = critical_path.nodes[lca_node].get('label', lca_node)
        print(f"   [FOUND] LCA / Potential Root Cause: {lca_label} (Hash: {lca_node})")

        # Thêm LCA vào verified_graph nếu nó chưa có (để hiển thị nguyên nhân gốc)
        if lca_node not in verified_graph:
            verified_graph.add_node(lca_node, label=lca_label)
            # Thêm các cạnh từ LCA đến các node trong verified_graph (nếu có trong critical_path)
            # Để nối LCA vào đồ thị hiển thị
            for target in list(verified_graph.nodes()):
                if target != lca_node:
                    # Tìm đường đi ngắn nhất từ LCA đến target trong critical_path để nối lại
                    try:
                        path = nx.shortest_path(critical_path, source=lca_node, target=target)
                        # Thêm đường đi này vào verified_graph để liền mạch
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i + 1]
                            u_l = critical_path.nodes[u].get('label', u)
                            v_l = critical_path.nodes[v].get('label', v)
                            verified_graph.add_node(u, label=u_l)
                            verified_graph.add_node(v, label=v_l)
                            verified_graph.add_edge(u, v)
                    except nx.NetworkXNoPath:
                        pass
    else:
        print("   [Info] LCA not found (Graph might be disjoint or too sparse).")

    # --- VISUALIZATION ---
    apply_visual_style(verified_graph)
    apply_visual_style(critical_path)
    apply_visual_style(summary_graph)

    # Highlight LCA Node
    if lca_node and lca_node in verified_graph:
        verified_graph.nodes[lca_node]['color'] = 'red'
        verified_graph.nodes[lca_node]['fontcolor'] = 'red'
        verified_graph.nodes[lca_node]['penwidth'] = '2.0'
        # Nếu muốn label rõ hơn:
        verified_graph.nodes[lca_node]['label'] = f"ROOT: {verified_graph.nodes[lca_node].get('label', '')}"

    print(f"Stats:")
    print(f" - Critical Path Edges: {critical_path.number_of_edges()}")
    print(f" - Summary Graph Edges: {summary_graph.number_of_edges()}")
    print(f" - VERIFIED ATTACK EDGES: {verified_graph.number_of_edges()}")

    if verified_graph.number_of_edges() > 0:
        # Xuất file ảnh (giữ nguyên logic phần trước tôi đã gửi)
        output_dot = f"{artifact_dir}verified_attack_path.dot"
        output_png = f"{artifact_dir}verified_attack_path.png"
        nx.drawing.nx_pydot.write_dot(verified_graph, output_dot)

        summary_dot = f"{artifact_dir}summary_graph.dot"
        summary_png = f"{artifact_dir}summary_graph.png"
        nx.drawing.nx_pydot.write_dot(summary_graph, summary_dot)

        critical_dot = f"{artifact_dir}critical_path.dot"
        critical_png = f"{artifact_dir}critical_path.png"
        nx.drawing.nx_pydot.write_dot(critical_path, critical_dot)

        try:
            os.system(f"dot -Tpng {output_dot} -o {output_png}")
            os.system(f"dot -Tpng {summary_dot} -o {summary_png}")
            os.system(f"dot -Tpng {critical_dot} -o {critical_png}")
            print(f"SUCCESS: Results saved to {output_png}")
        except:
            print("Saved dot files only.")
    else:
        print("No graph generated.")


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()