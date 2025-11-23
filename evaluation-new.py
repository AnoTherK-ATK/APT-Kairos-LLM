from sklearn.metrics import confusion_matrix
import logging

from kairos_utils import *
from config import *
from model import *
from sklearn.metrics import confusion_matrix, roc_auc_score

# --- SETUP LOGGING ---
logger = logging.getLogger("evaluation_logger")
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()

file_handler = logging.FileHandler(artifact_dir + 'evaluation.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def get_queue_anomaly_score(queue_list):
    """
    SỬA LOGIC 1: Chuyển từ tích sang tổng Log để tránh tràn số (Overflow)
    """
    score = 0
    for item in queue_list:
        # Log-Likelihood logic: log(loss + small_epsilon)
        score += np.log(item['loss'] + 1e-9)
    return score


def calculate_dynamic_threshold(history_path):
    """
    Tính ngưỡng động và IN RA THỐNG KÊ điểm của tập Validation
    """
    logger.info(f"--- Đang học ngưỡng từ: {history_path} ---")
    if not os.path.exists(history_path):
        logger.error("File history validation không tồn tại!")
        return 999999

    history_list = torch.load(history_path)
    scores = []

    # In ra điểm của các mẫu trong tập validation để kiểm tra
    logger.info("Danh sách điểm trong tập Validation (Benign):")
    for idx, hl in enumerate(history_list):
        score = get_queue_anomaly_score(hl)
        scores.append(score)
        # Chỉ in mẫu 5 cái đầu tiên để đỡ rối mắt, hoặc bỏ comment dòng dưới để in hết
        if idx < 5:
            logger.info(f"  Validation Queue {idx}: {score:.4f}")

    scores = np.array(scores)
    mu = np.mean(scores)
    sigma = np.std(scores)
    max_val = np.max(scores)

    # Ngưỡng = Mean + 3 * Std (Quy tắc thống kê)
    threshold = mu + 1.5 * sigma

    logger.info("-" * 20)
    logger.info(f"Thống kê Validation -> Mean: {mu:.4f} | Std: {sigma:.4f} | Max Score: {max_val:.4f}")
    logger.info(f"NGƯỠNG QUYẾT ĐỊNH (Threshold): {threshold:.4f}")
    logger.info("-" * 20)

    return threshold


def get_ground_truth(file_list):
    """
    SỬA LOGIC 3: Gán nhãn dựa trên khớp chuỗi thông minh hơn (Robust Matching)
    Thay vì khớp chính xác 100% tên file dài.
    """
    labels = {}
    # Các mốc thời gian tấn công (chỉ lấy phần đặc trưng nhất)
    # Dựa trên dataset DARPA E3
    attack_signatures = [
        "2018-04-06 11:18", "2018-04-06 11:33",
        "2018-04-06 11:48", "2018-04-06 12:03"
    ]

    for fname in file_list:
        is_attack = 0
        for sig in attack_signatures:
            if sig in fname:
                is_attack = 1
                break
        labels[fname] = is_attack
    return labels


def classifier_evaluation(y_test, y_test_pred):
    if len(set(y_test)) < 2:
        logger.warning("Ground truth chỉ có 1 lớp dữ liệu. AUC mặc định = 0.5")
        auc_val = 0.5
    else:
        try:
            auc_val = roc_auc_score(y_test, y_test_pred)
        except ValueError:
            auc_val = 0.5

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

    print("\n" + "=" * 30)
    print("KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG")
    print("=" * 30)
    logger.info(f'TP (Đúng tấn công): {tp}')
    logger.info(f'TN (Đúng bình thường): {tn}')
    logger.info(f'FP (Báo động giả): {fp}')
    logger.info(f'FN (Bỏ sót tấn công): {fn}')

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    fscore = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {fscore:.4f}")
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"AUC:       {auc_val:.4f}")

    return precision, recall, fscore, accuracy, auc_val

def calc10(y):
    cnt1 = 0
    cnt0 = 0
    for val in y:

        if y[val] == 1:
            cnt1 += 1
        else:
            cnt0 += 1
    return cnt1, cnt0


if __name__ == "__main__":
    logger.info("Bắt đầu quá trình đánh giá...")

    # --- BƯỚC 1: TÍNH NGƯỠNG (TRAINING PHASE) ---
    val_path = f"{artifact_dir}/graph_4_5_history_list"
    threshold = calculate_dynamic_threshold(val_path)

    # --- BƯỚC 2: ĐÁNH GIÁ (TESTING PHASE) ---
    pred_label = {}

    # Lấy danh sách toàn bộ file cần test
    all_test_files = []
    test_days = ['graph_4_6', 'graph_4_7']

    for day in test_days:
        dir_path = f"{artifact_dir}/{day}/"
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            all_test_files.extend(files)
            for f in files:
                pred_label[f] = 0  # Mặc định là bình thường

    logger.info(f"\n--- ĐANG QUÉT DỮ LIỆU TEST (Total: {len(all_test_files)} files) ---")

    # Duyệt qua lịch sử các queue đã xây dựng để tính điểm
    for day in test_days:
        hist_path = f"{artifact_dir}/{day}_history_list"
        if not os.path.exists(hist_path):
            continue

        history_list = torch.load(hist_path)

        for idx, hl in enumerate(history_list):
            # 1. Tính điểm
            anomaly_score = get_queue_anomaly_score(hl)

            # 2. IN RA ĐIỂM SỐ (Phần bạn yêu cầu thêm)
            # Xác định trạng thái để in log cho đẹp
            status = "NORMAL"
            if anomaly_score > threshold:
                status = "!!! ATTACK !!!"

            # In ra màn hình để theo dõi (Quan trọng cho khóa luận)
            logger.info(
                f"[{day}] Queue {idx:03d} | Score: {anomaly_score:.4f} | Threshold: {threshold:.4f} | Status: {status}")

            # 3. Quyết định gán nhãn
            if anomaly_score > threshold:
                queue_files = [item['name'] for item in hl]
                # Đánh dấu tất cả file trong queue này là tấn công
                for fname in queue_files:
                    pred_label[fname] = 1

    # --- BƯỚC 3: TÍNH TOÁN ĐỘ CHÍNH XÁC ---
    gt_labels = get_ground_truth(all_test_files)

    y_true = []
    y_pred = []

    # Chỉ so sánh những file có trong cả thư mục và ground truth
    valid_files = [f for f in all_test_files if f in gt_labels and f in pred_label]

    for fname in valid_files:
        y_true.append(gt_labels[fname])
        y_pred.append(pred_label[fname])

        # In ra các file bị phát hiện sai (False Positive / False Negative) để debug
        if gt_labels[fname] != pred_label[fname]:
            type_err = "FP (Báo giả)" if pred_label[fname] == 1 else "FN (Sót)"
            logger.warning(
                f"Sai lệch: {fname} -> Thực tế: {gt_labels[fname]}, Dự đoán: {pred_label[fname]} [{type_err}]")

    logger.info(f"\nSố lượng mẫu đánh giá hợp lệ: {len(y_true)}")
    classifier_evaluation(y_true, y_pred)