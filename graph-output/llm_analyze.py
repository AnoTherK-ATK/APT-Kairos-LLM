import os
import networkx as nx
import openai
# Fix warning typing như turn trước
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from google import genai
from google.genai import types
from typing import Optional, List, Union
from api_key import *


class GraphLLMAnalyzer:
    def __init__(self, openai_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None):
        self.openai_client = None
        self.gemini_model = None

        if openai_api_key:
            openai.api_key = openai_api_key
            self.openai_client = openai.Client(api_key=openai_api_key)

        if gemini_api_key:
            self.gemini_client = genai.Client(api_key=gemini_api_key)

    def load_graph_from_dot(self, dot_source: str, is_file_path: bool = True) -> nx.Graph:
        """
        Load đồ thị từ file .dot, giữ lại toàn bộ thuộc tính (bao gồm loss_score)
        """
        try:
            if is_file_path:
                if not os.path.exists(dot_source):
                    raise FileNotFoundError(f"Không tìm thấy file: {dot_source}")
                graphs = nx.drawing.nx_pydot.read_dot(dot_source)
            else:
                import pydot
                p_graphs = pydot.graph_from_dot_data(dot_source)
                graphs = nx.drawing.nx_pydot.from_pydot(p_graphs[0])
            return graphs
        except Exception as e:
            raise ValueError(f"Lỗi khi parse file DOT: {str(e)}")

    def _clean_and_summarize_graph(self, G: nx.Graph) -> str:
        """
        [CẬP NHẬT] Trích xuất thêm 'loss_score' từ thuộc tính cạnh.
        """
        description = []

        # Thống kê sơ bộ
        description.append(f"Đồ thị tấn công gồm {G.number_of_nodes()} thực thể và {G.number_of_edges()} hành vi.\n")

        # 1. Mô tả Node (Thêm thông tin type nếu có để context rõ hơn)
        description.append("--- DANH SÁCH THỰC THỂ (NODES) ---")
        for node, data in G.nodes(data=True):
            label = data.get('label', str(node)).replace('"', '')
            # Nếu node label quá dài (chứa cả IP, path...), có thể giữ nguyên vì LLM cần chi tiết này
            description.append(f"- ID: {node} | Label: {label}")

        # 2. Mô tả Cạnh (Hành vi + Mức độ bất thường)
        description.append("\n--- DANH SÁCH HÀNH VI (EDGES) & ĐIỂM BẤT THƯỜNG ---")

        # Sắp xếp cạnh theo loss_score giảm dần để LLM chú ý cái quan trọng trước
        edges_data = []
        for u, v, data in G.edges(data=True):
            src_label = G.nodes[u].get('label', str(u)).replace('"', '')
            dst_label = G.nodes[v].get('label', str(v)).replace('"', '')

            # Lấy các thuộc tính quan trọng
            action = data.get('label', 'unknown_action').replace('"', '')
            # Xử lý loss_score: trong file dot nó có thể là string
            loss_raw = data.get('loss_score', '0')
            try:
                loss_score = float(loss_raw)
            except:
                loss_score = 0.0

            time_raw = data.get('timestamp', '')  # Hoặc lấy từ label nếu label chứa time

            # Format dòng mô tả cho LLM
            # VD: "High Anomaly (5.89): cmd.exe -> runme.bat [EXECUTE]"
            edges_data.append({
                'text': f"- {src_label} --[{action}]--> {dst_label}",
                'loss': loss_score,
                'details': f"(Anomaly Score: {loss_score:.4f})"
            })

        # Sort: Điểm cao lên đầu
        edges_data.sort(key=lambda x: x['loss'], reverse=True)

        for item in edges_data:
            # Đánh dấu các cạnh rất cao để LLM chú ý
            prefix = "[CRITICAL] " if item['loss'] > 3.0 else ""
            description.append(f"{prefix}{item['text']} {item['details']}")

        return "\n".join(description)

    def analyze_with_llm(self,
                         dot_input: str,
                         instruction: str,
                         provider: str = 'gemini',
                         use_raw_dot: bool = False,
                         model_name: str = "gemini-2.0-flash") -> str:

        # 1. Xử lý dữ liệu
        if use_raw_dot:
            if os.path.exists(dot_input):
                with open(dot_input, 'r', encoding='utf-8') as f:
                    graph_content = f.read()
            else:
                graph_content = dot_input
        else:
            G = self.load_graph_from_dot(dot_input, is_file_path=os.path.exists(dot_input))
            graph_content = self._clean_and_summarize_graph(G)

        # 2. [CẬP NHẬT] Prompt nâng cao với chỉ dẫn về Loss Score
        full_prompt = f"""
        Bạn là một chuyên gia điều tra số (Digital Forensics) và phân tích mã độc APT.
        Nhiệm vụ của bạn là phân tích một "Provenance Graph" (đồ thị nguồn gốc dữ liệu) để giải thích một cuộc tấn công mạng.

        Dưới đây là dữ liệu đồ thị dạng văn bản đã được trích xuất từ hệ thống phát hiện bất thường (KAIROS):

        ============== DỮ LIỆU ĐỒ THỊ BẮT ĐẦU ==============
        {graph_content}
        ============== DỮ LIỆU ĐỒ THỊ KẾT THÚC ==============

        **CHÚ GIẢI QUAN TRỌNG:**
        - **Anomaly Score (loss_score):** Là điểm số tái tạo lỗi từ mô hình GNN. 
          - Giá trị này **CÀNG CAO** nghĩa là hành vi này **CÀNG BẤT THƯỜNG** và khả năng cao là bước tấn công chính.
          - Các hành vi có điểm thấp có thể là nhiễu hoặc hành vi nền (background activity).
        - **[CRITICAL]:** Đánh dấu các sự kiện có điểm bất thường rất cao, cần ưu tiên phân tích.

        **YÊU CẦU CỦA NGƯỜI DÙNG:**
        "{instruction}"

        **HƯỚNG DẪN TRẢ LỜI:**
        1. **Xác định trọng tâm:** Bắt đầu bằng việc phân tích các cạnh có `Anomaly Score` cao nhất. Chúng đóng vai trò gì trong chuỗi tấn công?
        2. **Dựng lại kịch bản (Storytelling):** Kết nối các sự kiện thành một câu chuyện có trình tự thời gian và logic (nguyên nhân - kết quả).
        3. **Kết luận:** Đưa ra nhận định về loại tấn công.

        Hãy trả lời bằng tiếng Việt chi tiết và mạch lạc.
        """

        # 3. Gọi API
        try:
            if provider.lower() == 'gemini':
                if not self.gemini_client: return "Lỗi: Thiếu Gemini Key."

                # [MỚI] Gọi API theo cú pháp Client mới của Google Gen AI SDK
                response = self.gemini_client.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                    # config=types.GenerateContentConfig(temperature=0.2) # Tùy chọn thêm config nếu cần
                )
                return response.text

            elif provider.lower() == 'chatgpt':
                if not self.openai_client: return "Lỗi: Thiếu OpenAI Key."

                messages_payload: List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]] = [
                    {"role": "system", "content": "Bạn là AI Security Analyst."},
                    {"role": "user", "content": full_prompt}
                ]
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o", messages=messages_payload, temperature=0.1
                )
                return response.choices[0].message.content

            else:
                return "Provider không hợp lệ."

        except Exception as e:
            return f"Lỗi API: {str(e)}"


if __name__ == "__main__":
    # 1. Cấu hình Key (Thay bằng key thật của bạn)


    analyzer = GraphLLMAnalyzer(openai_api_key=OPENAI_KEY, gemini_api_key=GEMINI_KEY)

    # 2. File dot đầu ra từ GNN Explainer
    dot_file_path = "verified_attack_path.dot"

    # Tạo file giả lập để test nếu chưa có file thật

    # 3. Instruction Prompt
    instruction_prompt = """
    Nhiệm vụ của bạn là phân tích đồ thị nguồn gốc chứa các node và edge liên quan đến một cuộc tấn công mạng. 
    Thực hiện phân tích, đối chiếu các dấu hiệu bất thường xuất hiện trong đồ thị có khả năng cao liên quan đến tấn công mạng (tên tiến trình bất thường, IP, port , các file nhạy cảm). 
    Từ đó cung cấp cho tôi các kịch bản tấn công có thể có từ những thông tin trong đồ thị. 
    Sau đó bằng với kinh nghiệm dày dặn của mình hãy chỉ ra những manh mối quan trọng, xâu chuỗi lại thành sơ đồ hành vi tấn công.
    Cuối cùng lập ra một báo cáo phân tích (dạng markdown và mermaid) về những gì mà bạn điều tra được.
    """

    # 4. Gọi Gemini với chế độ xử lý NetworkX (Cách tối ưu)
    print("--- Phân tích bởi Gemini ---")
    result = analyzer.analyze_with_llm(
        dot_input=dot_file_path,
        instruction=instruction_prompt,
        provider="gemini",
        model_name="gemini-3-pro-preview"  # Hoặc gemini-1.5-pro
    )
    with open("report.md", "w") as f:
        f.write(result)

