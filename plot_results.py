import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

# --- CẤU HÌNH ---
# Thư mục chứa các file log (kết quả chạy từ run_experiments.sh)

RESULTS_DIR = ("./experiment_results" if len(sys.argv) < 2 else sys.argv[1])


def parse_log_file(filepath):
    """
    Đọc file log với định dạng mới và trích xuất thông tin.
    Định dạng log: "YYYY-MM-DD HH:MM:SS - INFO - key: value"
    """
    data = {
        "Precision": 0.0,
        "Recall": 0.0,
        "F1-Score": 0.0,
        "Accuracy": 0.0,
        "AUC": 0.0,
        "Validation_Max_Score": 0.0,  # Tương đương ngưỡng (Threshold)
        "Avg_Attack_Score": 0.0  # Điểm trung bình của các cuộc tấn công phát hiện được
    }

    attack_scores = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            # 1. Trích xuất các chỉ số hiệu năng (Classification Metrics)
            # Log mẫu: ... - INFO - precision: 0.25
            if "precision:" in line:
                data["Precision"] = float(line.split("precision:")[-1].strip())

            elif "recall:" in line:
                data["Recall"] = float(line.split("recall:")[-1].strip())

            elif "fscore:" in line:
                data["F1-Score"] = float(line.split("fscore:")[-1].strip())

            elif "accuracy:" in line:
                data["Accuracy"] = float(line.split("accuracy:")[-1].strip())

            elif "auc_val:" in line:
                data["AUC"] = float(line.split("auc_val:")[-1].strip())

            # 2. Trích xuất ngưỡng (Dựa trên điểm validation cao nhất)
            # Log mẫu: ... The largest anomaly score in validation set is: 13.79...
            elif "largest anomaly score in validation set is:" in line:
                val_part = line.split("is:")[-1].strip()
                data["Validation_Max_Score"] = float(val_part)

            # 3. Trích xuất điểm bất thường của các tấn công phát hiện được
            # Log mẫu: ... - INFO - Anomaly score: 146.24...
            elif "Anomaly score:" in line and "largest" not in line:
                score_part = line.split("Anomaly score:")[-1].strip()
                attack_scores.append(float(score_part))

        # Tính điểm trung bình của các tấn công (để so sánh scale)
        if attack_scores:
            data["Avg_Attack_Score"] = np.mean(attack_scores)

    except Exception as e:
        print(f"Lỗi khi đọc file {filepath}: {e}")

    return data


def plot_classification_metrics(df):
    """
    Vẽ biểu đồ so sánh các chỉ số: Precision, Recall, F1, Accuracy, AUC
    """
    if df.empty:
        print("Không có dữ liệu để vẽ biểu đồ Metrics.")
        return

    metrics = ["Precision", "Recall", "F1-Score", "Accuracy", "AUC"]
    models = df.index

    x = np.arange(len(models))  # Vị trí các nhóm
    width = 0.15  # Độ rộng mỗi cột

    fig, ax = plt.subplots(figsize=(12, 6))

    # Vẽ từng metric
    for i, metric in enumerate(metrics):
        offset = width * i
        # Lấy giá trị cột, nếu không có thì mặc định 0
        values = df.get(metric, [0] * len(models))
        rects = ax.bar(x + offset, values, width, label=metric)
        # Ghi số lên đầu cột
        ax.bar_label(rects, padding=2, fmt='%.2f', fontsize=7)

    # Căn chỉnh biểu đồ
    ax.set_ylabel('Giá trị (0.0 - 1.0)')
    ax.set_title('So sánh Hiệu năng Phát hiện Tấn công (Metrics)')

    # Đặt nhãn trục X vào giữa nhóm cột
    center_offset = width * (len(metrics) - 1) / 2
    ax.set_xticks(x + center_offset)
    ax.set_xticklabels([m.upper() for m in models])

    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0))
    ax.set_ylim(0, 1.15)  # Tăng chiều cao để chứa nhãn số
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    save_path = os.path.join(RESULTS_DIR, "comparison_metrics.png")
    plt.savefig(save_path)
    print(f"[graph] Đã lưu biểu đồ Metrics tại: {save_path}")
    plt.close()  # Đóng figure để giải phóng bộ nhớ


def plot_anomaly_scale(df):
    """
    Vẽ biểu đồ so sánh thang đo (Scale) giữa điểm Validation (Bình thường) và Attack (Tấn công).
    Biểu đồ này chứng minh luận điểm: GraphSAGE có điểm cao hơn nhưng khoảng cách phân tách vẫn rõ ràng.
    """
    if df.empty: return

    models = df.index
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # Cột 1: Điểm Validation cao nhất (Đại diện cho ngưỡng nền)
    rects1 = ax.bar(x - width / 2, df['Validation_Max_Score'], width,
                    label='Max Benign Score (Threshold Ref)', color='green', alpha=0.7)

    # Cột 2: Điểm Tấn công trung bình
    rects2 = ax.bar(x + width / 2, df['Avg_Attack_Score'], width,
                    label='Avg Attack Score', color='red', alpha=0.7)

    ax.set_ylabel('Anomaly Score (Log Scale)')
    ax.set_title('So sánh Thang đo Điểm Bất thường (Benign vs Attack)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()

    # Gán nhãn giá trị (làm tròn số nguyên cho gọn vì điểm thường lớn)
    ax.bar_label(rects1, padding=3, fmt='%.0f', fontsize=9)
    ax.bar_label(rects2, padding=3, fmt='%.0f', fontsize=9)

    # Chuyển trục Y sang logarit nếu chênh lệch quá lớn (VD: 10 vs 10000)
    # ax.set_yscale('log')

    save_path = os.path.join(RESULTS_DIR, "comparison_scale.png")
    plt.savefig(save_path)
    print(f"[graph] Đã lưu biểu đồ Scale tại: {save_path}")
    plt.close()


def main():
    print("--- ĐANG PHÂN TÍCH KẾT QUẢ TỪ LOG ---")

    if not os.path.exists(RESULTS_DIR):
        print(f"Lỗi: Thư mục '{RESULTS_DIR}' không tồn tại.")
        return

    # Tìm tất cả file .log trong thư mục kết quả
    log_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".log")]

    if not log_files:
        print(f"Cảnh báo: Không tìm thấy file .log nào trong {RESULTS_DIR}.")
        return

    results = {}
    print(f"Tìm thấy {len(log_files)} file log: {log_files}")

    for filename in log_files:
        # Tách tên mô hình từ tên file (VD: evaluation_sage.log -> sage)
        model_name = filename.replace("evaluation_", "").replace(".log", "")
        file_path = os.path.join(RESULTS_DIR, filename)

        # Parse dữ liệu
        results[model_name] = parse_log_file(file_path)

    # Chuyển sang Pandas DataFrame
    df = pd.DataFrame(results).T  # Transpose để dòng là Model

    # Sắp xếp lại thứ tự nếu có (để UniMP lên đầu làm chuẩn)
    priority_order = ['unimp', 'gat', 'rgcn', 'sage', 'gcn']
    existing_models = [m for m in priority_order if m in df.index]
    other_models = [m for m in df.index if m not in existing_models]
    df = df.reindex(existing_models + other_models)

    print("\n--- BẢNG TỔNG HỢP SỐ LIỆU ---")
    print(df)

    # Lưu bảng số liệu ra CSV
    csv_path = os.path.join(RESULTS_DIR, "summary_results.csv")
    df.to_csv(csv_path)
    print(f"\n[csv] Đã lưu bảng số liệu tại: {csv_path}")

    # Vẽ biểu đồ
    plot_classification_metrics(df)
    plot_anomaly_scale(df)

    print("\n=== HOÀN TẤT ===")


if __name__ == "__main__":
    main()