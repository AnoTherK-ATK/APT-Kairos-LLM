import networkx as nx
import json
import logging
from typing import List, Dict, Union

# Import từ thư viện GoT
from graph_of_thoughts import controller, language_models, operations, prompter, parser
from graph_of_thoughts.controller import Controller
from graph_of_thoughts.operations import GraphOfOperations
from graph_of_thoughts.language_models import Llama2HF
from config import artifact_dir


# ==========================================
# 1. PARSER (Bắt buộc phải implement Abstract Class này)
# ==========================================
class APTParser(parser.Parser):
    """
    Phân tích phản hồi từ LLM để chuyển thành dữ liệu có cấu trúc.
    """

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        # Phân tích các giả thuyết tấn công do LLM sinh ra
        new_states = []
        for text in texts:
            new_state = state.copy()
            new_state["current_path"] = text.strip()
            new_states.append(new_state)
        return new_states

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        # Phân tích điểm số (0-10) từ phản hồi của LLM
        scores = []
        for text in texts:
            # Tìm số đầu tiên trong chuỗi phản hồi (VD: "Score: 9/10" -> lấy 9)
            try:
                score = float(''.join(filter(str.isdigit, text.split('/')[0])))
                # Chuẩn hóa về thang 0-1 nếu cần, ở đây giữ nguyên
            except ValueError:
                score = 0.0  # Nếu không tìm thấy số, gán 0
            scores.append(score)
        return scores

    def parse_aggregation_answer(self, states: List[Dict], texts: List[str]) -> Union[Dict, List[Dict]]:
        # Gộp các path thành một critical path duy nhất
        return {"final_critical_path": texts[0]}

    # Các hàm này chưa dùng đến trong flow đơn giản nhưng vẫn phải khai báo để không lỗi Abstract Class
    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        return state

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        return True


# ==========================================
# 2. PROMPTER (Tùy chỉnh cho APT)
# ==========================================
class APTPrompter(prompter.Prompter):
    def generate_prompt(self, num_branches: int, data_interactions: str, **kwargs) -> str:
        return f"""
        [System]: You are a cybersecurity expert analyzing a provenance graph.
        [Input]: The following system interactions were detected:
        {data_interactions}

        [Task]: Identify {num_branches} distinct potential attack sequences (suspicious chains of events). 
        Focus on interactions involving 'wget', 'chmod', 'insmod', or suspicious IPs.
        output strictly the sequence.
        """

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        # Lấy danh sách các path cần chấm điểm
        paths_text = "\n".join([f"Path {i}: {s['current_path']}" for i, s in enumerate(state_dicts)])
        return f"""
        Rate the suspiciousness of the following attack paths from 0 to 10 (10 = Confirmed APT).

        {paths_text}

        Output strictly in format: 
        Path 0: <score>
        Path 1: <score>
        ...
        """

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        top_paths = "\n".join([s['current_path'] for s in state_dicts])
        return f"""
        Synthesize the following suspicious paths into a single "Critical Attack Path" narrative.

        {top_paths}

        Final Narrative:
        """

    # Dummy implementations
    def improve_prompt(self, **kwargs) -> str: return ""

    def validation_prompt(self, **kwargs) -> str: return ""


# ==========================================
# 3. XÂY DỰNG GRAPH OF OPERATIONS
# ==========================================
def build_apt_operations_graph() -> GraphOfOperations:
    """
    Quy trình: Generate (Sinh giả thuyết) -> Score (Chấm điểm) -> KeepBest (Lọc) -> Aggregate (Tổng hợp)
    """
    ops_graph = GraphOfOperations()

    # Bước 1: Sinh ra 3 luồng suy nghĩ (giả thuyết tấn công) từ dữ liệu gốc
    gen_op = operations.Generate(1, 3)
    ops_graph.append_operation(gen_op)

    # Bước 2: Chấm điểm các giả thuyết này
    score_op = operations.Score(1, True)  # True = combined scoring (chấm cùng lúc cho nhanh)
    ops_graph.append_operation(score_op)

    # Bước 3: Giữ lại 1 giả thuyết tốt nhất (điểm cao nhất)
    keep_best_op = operations.KeepBestN(1, True)
    ops_graph.append_operation(keep_best_op)

    # Bước 4: Tổng hợp (Aggregate) - Nếu giữ lại >1 thì bước này sẽ gộp lại.
    # Ở đây giữ 1 nên nó sẽ format lại kết quả cuối cùng.
    agg_op = operations.Aggregate(1)
    ops_graph.append_operation(agg_op)

    return ops_graph


# ==========================================
# 4. HÀM XỬ LÝ DỮ LIỆU ĐẦU VÀO
# ==========================================
def parse_dot_file(dot_path):
    # Đọc file dot đơn giản
    import re
    interactions = []
    try:
        with open(dot_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "->" in line:
                    # Làm sạch chuỗi
                    line = line.replace('"', '').replace(';', '').strip()
                    parts = line.split('->')
                    if len(parts) == 2:
                        interactions.append(f"{parts[0].strip()} interacts with {parts[1].strip()}")
    except Exception as e:
        print(f"Error reading dot file: {e}")
        return "Simulation Data: Process A downloaded File B"  # Fallback nếu lỗi

    # Chỉ lấy 20 interaction đầu tiên để tránh quá tải Prompt
    return "\n".join(interactions[:30])


# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Config Model (Dùng Llama HF hoặc ChatGPT/Gemini giả lập)
    # Lưu ý: Cần chỉnh đường dẫn model_name đúng với máy bạn
    lm = Llama2HF(
        config_path="./config.json",  # Tạo file này nếu chưa có
        model_name="llama7b-hf",
        cache=True
    )

    # 2. Chuẩn bị dữ liệu
    dot_file_path = f"{artifact_dir}w04-t15.dot"  # File dot bạn đã upload
    data_str = parse_dot_file(dot_file_path)

    # Dữ liệu ban đầu truyền vào Controller
    initial_state = {
        "data_interactions": data_str,
        "current_path": ""
    }

    # 3. Khởi tạo các thành phần
    apt_graph = build_apt_operations_graph()
    apt_prompter = APTPrompter()
    apt_parser = APTParser()  # <--- Đây là cái bạn thiếu trong lỗi cũ

    # 4. Khởi tạo Controller (Đúng 5 tham số)
    controller_instance = Controller(
        lm,  # 1. Language Model
        apt_graph,  # 2. Graph of Operations
        apt_prompter,  # 3. Prompter
        apt_parser,  # 4. Parser
        initial_state  # 5. Problem Parameters
    )

    # 5. Chạy
    print(">>> Starting Graph of Thoughts for APT Detection...")
    controller_instance.run()

    # 6. Xuất kết quả
    controller_instance.output_graph("apt_analysis_result.json")
    print(">>> Done! Result saved to apt_analysis_result.json")