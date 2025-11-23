import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Cấu hình thư mục chứa log
RESULTS_DIR = "./experiment_results"


def parse_log_file(filepath):
    """
    Đọc file log và trích xuất các chỉ số quan trọng bằng Regex
    """
    data = {
        "Precision": 0,
        "Recall": 0,
        "F1-Score": 0,
        "Accuracy": 0,
        "Threshold": 0,
        "Avg_Anomaly_Score": 0
    }

    scores = []

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.readlines()

    for line in content:
        # 1. Lấy các chỉ số phân loại (Classifier Metrics)
        if "Precision:" in line:
            data["Precision"] = float(re.search(r"Precision:\s+(\d+\.\d+)", line).group(1))
        elif "Recall:" in line:
            data["Recall"] = float(re.search(r"Recall:\s+(\d+\.\d+)", line).group(1))
        elif "F1-Score:" in line:
            data["F1-Score"] = float(re.search(r"F1-Score:\s+(\d+\.\d+)", line).group(1))
        elif "Accuracy:" in line:
            data["Accuracy"] = float(re.search(r"Accuracy:\s+(\d+\.\d+)", line).group(1))

        # 2. Lấy ngưỡng (Threshold) - đại diện cho độ lớn (scale) của model
        elif "NGƯỠNG QUYẾT ĐỊNH (Threshold):" in line:
            data["Threshold"] = float(re.search(r"Threshold\):\s+(\d+\.\d+)", line).group(1))

        # 3. Thu thập điểm Anomaly Score của từng Queue để tính trung bình
        # Mẫu: Queue 003 | Score: 1234.5678 | ...
        elif "| Score:" in line:
            match = re.search(r"Score:\s+(\d+\.\d+)", line)
            if match:
                scores.append(float(match.group(1)))

    if scores:
        data["Avg_Anomaly_Score"] = np.mean(scores)

    return data


def plot_classification_metrics(df):
    """
    Vẽ biểu đồ so sánh hiệu năng phát hiện
    """
    models = df.index
    metrics = ["Precision", "Recall", "F1-Score", "Accuracy"]

    x = np.arange(len(models))  # Vị trí các nhãn
    width = 0.2  # Độ rộng cột

    fig, ax = plt.subplots(figsize=(10, 6))

    # Vẽ từng nhóm cột
    for i, metric in enumerate(metrics):
        offset = width * i
        rects = ax.bar(x + offset, df[metric], width, label=metric)
        # Thêm số liệu lên đầu cột
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=8)

    ax.set_ylabel('Điểm số (0.0 - 1.0)')
    ax.set_title('So sánh Hiệu năng Phát hiện Tấn công giữa các Mô hình GNN')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)  # Để chừa chỗ cho label

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/comparison_metrics.png")
    print(f"Đã lưu biểu đồ hiệu năng: {RESULTS_DIR}/comparison_metrics.png")
    plt.show()


def plot_anomaly_scale(df):
    """
    Vẽ biểu đồ so sánh độ lớn của Anomaly Score (Scale)
    Đây là biểu đồ quan trọng để chứng minh luận điểm về GraphSAGE vs UniMP
    """
    models = df.index

    fig, ax = plt.subplots(figsize=(8, 5))

    # Vẽ cột Threshold
    bars = ax.bar(models, df["Threshold"], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)])

    ax.set_ylabel('Giá trị Ngưỡng (Log-Likelihood Scale)')
    ax.set_title('So sánh Thang đo Điểm Bất thường (Anomaly Score Scale)')
    ax.bar_label(bars, fmt='%.0f')

    # Thêm chú thích
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = '\n'.join((
        'Nhận xét:',
        '- Scale điểm khác nhau do kiến trúc GNN.',
        '- GraphSAGE/GCN thường có điểm cao hơn',
        '  do thiếu thông tin ngữ nghĩa cạnh (Edge Attr).',
        '- Tuy nhiên, hiệu năng phát hiện vẫn tốt.'
    ))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/comparison_scale.png")
    print(f"Đã lưu biểu đồ thang đo: {RESULTS_DIR}/comparison_scale.png")
    plt.show()


def main():
    # 1. Quét file log
    results = {}
    if not os.path.exists(RESULTS_DIR):
        print(f"Thư mục {RESULTS_DIR} không tồn tại. Hãy chạy run_experiments.sh trước.")
        return

    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".log")]

    if not files:
        print("Không tìm thấy file log nào.")
        return

    print(f"Tìm thấy {len(files)} file log. Đang phân tích...")

    for filename in files:
        # Tên file dạng evaluation_unimp.log -> lấy 'unimp'
        model_name = filename.replace("evaluation_", "").replace(".log", "")
        filepath = os.path.join(RESULTS_DIR, filename)

        data = parse_log_file(filepath)
        results[model_name] = data

    # 2. Chuyển sang DataFrame để dễ vẽ
    df = pd.DataFrame(results).T  # Transpose để dòng là model

    # Sắp xếp lại thứ tự index cho đẹp (nếu có đủ model)
    desired_order = ['unimp', 'gat', 'rgcn', 'sage', 'gcn']
    # Chỉ giữ lại những cái có trong df
    existing_order = [m for m in desired_order if m in df.index]
    # Thêm những cái khác nếu có
    remaining = [m for m in df.index if m not in existing_order]
    df = df.reindex(existing_order + remaining)

    print("\n--- BẢNG TỔNG HỢP KẾT QUẢ ---")
    print(df)

    # Xuất ra Excel/CSV để đưa vào báo cáo
    df.to_csv(f"{RESULTS_DIR}/summary_results.csv")

    # 3. Vẽ biểu đồ
    plot_classification_metrics(df)
    plot_anomaly_scale(df)


if __name__ == "__main__":
    main()