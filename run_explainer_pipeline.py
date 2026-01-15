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
from llm_analyze import *
from api_key import *

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


import re


def get_shape_from_label(label: str) -> str:
    """
    Phân loại nhãn (label) để trả về hình dạng tương ứng.
    - IPv4/IPv6: diamond
    - File path: ellipse
    - Khác (Process): box
    """

    # 1. Regex cho IPv4 (Chính xác hơn: giới hạn số từ 0-255)
    # Rút gọn: \d{1,3} lặp lại 3 lần kèm dấu chấm, kết thúc bằng \d{1,3}
    ipv4_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'

    # 2. Regex cho File Path
    # Bắt đầu bằng / (Linux), ./ hoặc ../ (Relative), hoặc C:\ (Windows)
    file_pattern = r'^(\/|\.\.?\/|[a-zA-Z]:\\)'

    # Kiểm tra IP (Ưu tiên IPv4 hoặc chứa dấu ':' và số cho IPv6)
    is_ip = re.search(ipv4_pattern, label) or (':' in label and any(c.isdigit() for c in label))

    if is_ip:
        return 'diamond'

    # Kiểm tra File Path
    if re.match(file_pattern, label):
        return 'ellipse'

    # Mặc định là Process
    return 'box'


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


def find_multiple_lcas(graph, nodes_of_interest):
    """
    Tìm tập hợp các LCA tối thiểu để bao phủ tất cả nodes_of_interest.
    Chiến lược: Greedy Set Cover kết hợp với tìm LCA cục bộ.
    """
    if not nodes_of_interest:
        return []

    # Chỉ xét các node có trong graph
    target_nodes = set(n for n in nodes_of_interest if n in graph)
    if not target_nodes:
        return []

    lca_roots = []
    uncovered = target_nodes.copy()

    while uncovered:
        # 1. Đếm số lượng node mục tiêu mà mỗi tổ tiên có thể bao phủ
        candidate_counts = {}

        # Duyệt qua từng node chưa được cover để tìm tổ tiên của nó
        for node in uncovered:
            # Lấy tất cả tổ tiên và chính nó
            ancestors = nx.ancestors(graph, node)
            ancestors.add(node)

            for anc in ancestors:
                candidate_counts[anc] = candidate_counts.get(anc, 0) + 1

        if not candidate_counts:
            # Trường hợp đồ thị bị rời rạc hoàn toàn, không tìm thấy tổ tiên nào chung
            # Fallback: Lấy chính các node còn lại làm root
            print(f"   [Info] Cannot find common ancestors for remaining {len(uncovered)} nodes.")
            lca_roots.extend(list(uncovered))
            break

        # 2. Chọn ứng viên bao phủ được nhiều node nhất (Greedy)
        best_candidate = max(candidate_counts, key=candidate_counts.get)

        # 3. Xác định cụm node được bao phủ bởi ứng viên này
        # (Để tìm LCA chính xác thấp nhất cho cụm này, thay vì lấy best_candidate có thể ở quá cao)
        covered_subset = set()
        for node in uncovered:
            if node == best_candidate or best_candidate in nx.ancestors(graph, node):
                covered_subset.add(node)

        # 4. Tìm LCA thấp nhất cho riêng cụm này (Refinement)
        # Hàm find_lca_for_set đã có sẵn trong code của bạn
        cluster_lca = find_lca_for_set(graph, covered_subset)

        if cluster_lca:
            lca_roots.append(cluster_lca)
        else:
            # Nếu không tìm được thấp hơn, dùng chính best_candidate
            lca_roots.append(best_candidate)

        # 5. Loại bỏ các node đã xử lý khỏi danh sách cần tìm
        uncovered -= covered_subset

    return lca_roots


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

        data['shape'] = get_shape_from_label(label)

    for u, v, data in graph.edges(data=True):
        data['color'] = 'blue'
        data['fontcolor'] = 'black'
        data['fontsize'] = '6'  # [MODIFIED] Giảm xuống 6 hoặc 7 cho gọn
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
                    src_ids, dst_ids, weights, types = explainer.explain_edge(
                        event['srcnode'], event['dstnode'], event['time'],
                        full_data, memory, neighbor_loader
                    )

                    timestamp = ns_time_to_datetime_US(event['time'])

                    for i in range(len(src_ids)):
                        u_id, v_id, w, t_idx = src_ids[i], dst_ids[i], weights[i], types[i]
                        t_idx_int = int(t_idx)
                        edge_type_str = rel2id.get(t_idx_int, f"Type_{t_idx_int}")
                        # [FIX] Sử dụng hàm thống nhất get_node_data
                        u_hash, u_label = get_node_data(u_id, nodeid2msg)
                        v_hash, v_label = get_node_data(v_id, nodeid2msg)

                        add_node_with_label(critical_path, u_hash, u_label)
                        add_node_with_label(critical_path, v_hash, v_label)

                        if w > 0.5:
                            edge_display_label = (
                                f"Time: {timestamp}\n"
                                # f"Type: {edge_type_str}\n"
                                f"Loss: {event['loss']:.4f}"
                            )
                            critical_path.add_edge(u_hash, v_hash,
                                                   type=edge_type_str,
                                                   loss_score=event['loss'],
                                                   label=edge_display_label)
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

                    timestamp = ns_time_to_datetime_US(event['time'])
                    edge_display_label = (
                        f"Time: {timestamp}\n"
                        f"Type: {event["edge_type"]}\n"
                        f"Loss: {event['loss']:.4f}"
                    )
                    add_node_with_label(louvain_input_graph, u_hash, u_label)
                    add_node_with_label(louvain_input_graph, v_hash, v_label)
                    louvain_input_graph.add_edge(u_hash, v_hash,
                                                 type=event["edge_type"],
                                                 loss_score=event['loss'],
                                                 label=edge_display_label)

        # --- 3. RUN LOUVAIN & UPDATE SUMMARY GRAPH ---
                # --- 3. RUN LOUVAIN & UPDATE SUMMARY GRAPH ---
            if louvain_input_graph.number_of_edges() > 0:
                undirected_g = louvain_input_graph.to_undirected()
                try:
                    # print(f"   Running Louvain on {louvain_input_graph.number_of_edges()} edges...")
                    partition = attack_investigation.community_louvain.best_partition(undirected_g)

                    # Chỉ thêm cạnh vào summary_graph nếu 2 node cùng community
                    for u, v, attr in louvain_input_graph.edges.data():
                        edge_label = attr.get('label')
                        edge_type = attr.get('type')
                        edge_weight = attr.get('loss_score')
                        if u in partition and v in partition:
                            if partition[u] == partition[v]:  # Cùng cộng đồng -> Giữ lại
                                summary_graph.add_edge(u, v,
                                                       type=edge_type,
                                                       loss_score=edge_weight,
                                                       label=edge_label)

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
    for u, v, attr in verified_graph_struct.edges.data():
        # Lấy label từ critical_path (nơi chứa label chính xác nhất)
        u_lbl = critical_path.nodes[u].get('label', u)
        v_lbl = critical_path.nodes[v].get('label', v)

        edge_data = summary_graph.get_edge_data(u, v)
        edge_label = edge_data.get('label', '') if edge_data else ''

        verified_graph.add_node(u, label=u_lbl)
        verified_graph.add_node(v, label=v_lbl)

        if summary_graph.has_edge(u, v):
            edge_data = summary_graph.get_edge_data(u, v)

            # Lấy các thông tin quan trọng
            w_val = edge_data.get('loss_score', 0.0)
            type_val = edge_data.get('type', 'unknown')
            label_val = edge_data.get('label', '')  # Đây là cái label chứa timestamp và loss ta đã tạo

            verified_graph.add_edge(u, v,
                                    loss_score=w_val,
                                    type=type_val,
                                    label=label_val)

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
    lca_nodes = find_multiple_lcas(summary_graph, target_nodes)

    lca_label = "Unknown"
    if lca_nodes:
        print(f"   [FOUND] Identified {len(lca_nodes)} Root Cause(s).")

        for idx, lca_node in enumerate(lca_nodes):
            lca_label = summary_graph.nodes[lca_node].get('label', lca_node)
            print(f"    - Root {idx + 1}: {lca_label} (Hash: {lca_node})")

            # Thêm LCA vào verified_graph để hiển thị
            if lca_node not in verified_graph:
                verified_graph.add_node(lca_node, label=lca_label)

            # Nối LCA với các node trong đồ thị (Tái tạo đường đi tấn công)
            # Chỉ nối tới những node chưa có cha nào khác trong verified_graph để đỡ rối,
            # hoặc nối tới tất cả target thuộc nhánh của nó.
            for target in list(verified_graph.nodes()):
                if target == lca_node: continue

                # Kiểm tra xem target có phải là hậu duệ của LCA này không
                try:
                    if nx.has_path(summary_graph, lca_node, target):
                        path = nx.shortest_path(summary_graph, source=lca_node, target=target)

                        # Thêm đường đi vào đồ thị
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i + 1]
                            # Copy thông tin node
                            if u not in verified_graph:
                                u_lbl = summary_graph.nodes[u].get('label', u)
                                verified_graph.add_node(u, label=u_lbl)
                            if v not in verified_graph:
                                v_lbl = summary_graph.nodes[v].get('label', v)
                                verified_graph.add_node(v, label=v_lbl)

                            # Copy thông tin cạnh (bao gồm timestamp label nếu bạn đã làm bước trước)
                            edge_data = summary_graph.get_edge_data(u, v)
                            edge_type = edge_data.get('type', 'unknown')
                            edge_loss_score = edge_data.get('loss_score', 0.0)
                            edge_label = edge_data.get('label', '') if edge_data else ''

                            # Tránh thêm cạnh trùng lặp
                            if not verified_graph.has_edge(u, v):
                                verified_graph.add_edge(u, v,
                                                        type=edge_type,
                                                        loss_score=edge_loss_score,
                                                        label=edge_label,
                                                        fontsize=6,
                                                        color='blue')
                except Exception:
                    pass


    else:
        print("   [Info] LCA not found (Graph might be too disjoint).")

    apply_visual_style(verified_graph)
    apply_visual_style(critical_path)
    apply_visual_style(summary_graph)
    for it, lca_node in enumerate(lca_nodes):
        if lca_node and lca_node in verified_graph:
            verified_graph.nodes[lca_node]['color'] = 'red'
            verified_graph.nodes[lca_node]['fontcolor'] = 'red'
            verified_graph.nodes[lca_node]['penwidth'] = '1.0'
            # Nếu muốn label rõ hơn:
            verified_graph.nodes[lca_node]['label'] = f"{verified_graph.nodes[lca_node].get('label', '')}"
    print(f"Stats:")
    print(f" - Critical Path Edges: {critical_path.number_of_edges()}")
    print(f" - Summary Graph Edges: {summary_graph.number_of_edges()}")
    print(f" - VERIFIED ATTACK EDGES: {verified_graph.number_of_edges()}")

    if verified_graph.number_of_edges() > 0:
        # Xuất file ảnh (giữ nguyên logic phần trước tôi đã gửi)
        output_dot = f"{artifact_dir}verified_attack_path.dot"
        output_png = f"{artifact_dir}verified_attack_path.pdf"
        nx.drawing.nx_pydot.write_dot(verified_graph, output_dot)

        summary_dot = f"{artifact_dir}summary_graph.dot"
        summary_png = f"{artifact_dir}summary_graph.pdf"
        nx.drawing.nx_pydot.write_dot(summary_graph, summary_dot)

        critical_dot = f"{artifact_dir}critical_path.dot"
        critical_png = f"{artifact_dir}critical_path.pdf"
        nx.drawing.nx_pydot.write_dot(critical_path, critical_dot)

        try:
            os.system(f"dot -Tpdf {output_dot} -o {output_png}")
            os.system(f"dot -Tpdf {summary_dot} -o {summary_png}")
            os.system(f"dot -Tpdf {critical_dot} -o {critical_png}")
            print(f"SUCCESS: Results saved to {output_png}")
        except:
            print("Saved dot files only.")
    else:
        print("No graph generated.")

    print("========Calling LLM to analyze=========")
    instruction_prompt = """
        Nhiệm vụ của bạn là phân tích đồ thị nguồn gốc chứa các node và edge liên quan đến một cuộc tấn công mạng. 
        Thực hiện phân tích, đối chiếu các dấu hiệu bất thường xuất hiện trong đồ thị có khả năng cao liên quan đến tấn công mạng (tên tiến trình bất thường, IP, port , các file nhạy cảm). 
        Từ đó cung cấp cho tôi các kịch bản tấn công có thể có từ những thông tin trong đồ thị. 
        Sau đó bằng với kinh nghiệm dày dặn của mình hãy chỉ ra những manh mối quan trọng, xâu chuỗi lại thành sơ đồ hành vi tấn công.
        Cuối cùng lập ra một báo cáo phân tích (dạng markdown và mermaid) về những gì mà bạn điều tra được.
        """

    analyzer = GraphLLMAnalyzer(openai_api_key=OPENAI_KEY, gemini_api_key=GEMINI_KEY)
    # dot_file_path = f"{artifact_dir}verified_attack_path.dot"
    result = analyzer.analyze_with_llm(
        instruction=instruction_prompt,
        provider="gemini",
        model_name="gemini-3-pro-preview",  # Hoặc gemini-1.5-pro
        use_graph_directive = True,
        graph_input=verified_graph,
    )
    with open("report.md", "w") as f:
        f.write(result)


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()