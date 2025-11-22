from sklearn.metrics import confusion_matrix
import logging

from kairos_utils import *
from config import *
from model import *
from sklearn.metrics import confusion_matrix, roc_auc_score

# Setting for logging
logger = logging.getLogger("evaluation_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + 'evaluation.log')
file_handler.setLevel(logging.INFO)
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
    SỬA LOGIC 2: Tính ngưỡng động dựa trên thống kê của tập Validation (Clean data)
    Thay vì dùng beta_day6 hardcode.
    """
    logger.info(f"Đang học ngưỡng từ tập Validation: {history_path}")
    history_list = torch.load(history_path, weights_only=False)
    scores = []
    for hl in history_list:
        scores.append(get_queue_anomaly_score(hl))

    scores = np.array(scores)
    mu = np.mean(scores)
    sigma = np.std(scores)

    # Quy tắc 3-Sigma: Bất cứ cái gì lệch quá 3 lần độ lệch chuẩn là bất thường
    threshold = mu + 3 * sigma
    logger.info(f"Validation Mean: {mu:.4f}, Std: {sigma:.4f}")
    logger.info(f"-> Calculated Threshold: {threshold:.4f}")
    return threshold


def get_ground_truth_robust(file_list):
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
    tn, fp, fn, tp =confusion_matrix(y_test, y_test_pred).ravel()
    logger.info(f'tn: {tn}')
    logger.info(f'fp: {fp}')
    logger.info(f'fn: {fn}')
    logger.info(f'tp: {tp}')

    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    fscore=2*(precision*recall)/(precision+recall)
    auc_val=roc_auc_score(y_test, y_test_pred)
    logger.info(f"precision: {precision}")
    logger.info(f"recall: {recall}")
    logger.info(f"fscore: {fscore}")
    logger.info(f"accuracy: {accuracy}")
    logger.info(f"auc_val: {auc_val}")
    return precision,recall,fscore,accuracy,auc_val

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
    logger.info("Start Evaluation...")

    # --- BƯỚC 1: HỌC NGƯỠNG (Thresholding) ---
    # Dùng dữ liệu ngày 5 (Validation - không có tấn công) để xác định thế nào là "bình thường"
    val_path = f"{artifact_dir}/graph_4_5_history_list"
    threshold = calculate_dynamic_threshold(val_path)

    # --- BƯỚC 2: ĐÁNH GIÁ (Testing) ---
    pred_label = {}
    all_test_files = []

    # Duyệt qua các ngày test (Ngày 6 và 7)
    for day in ['graph_4_6', 'graph_4_7']:
        # Lấy danh sách file thực tế trong thư mục
        current_files = os.listdir(f"{artifact_dir}/{day}/")
        all_test_files.extend(current_files)

        # Mặc định đoán là 0 (Bình thường)
        for f in current_files:
            pred_label[f] = 0

        # Load kết quả chạy từ model
        hist_path = f"{artifact_dir}/{day}_history_list"
        if os.path.exists(hist_path):
            history_list = torch.load(hist_path, weights_only=False)

            for hl in history_list:
                score = get_queue_anomaly_score(hl)

                # Logic so sánh ngưỡng
                if score > threshold:
                    # Nếu queue này bất thường, đánh dấu TẤT CẢ file trong đó là 1
                    for item in hl:
                        fname = item['name']
                        if fname in pred_label:
                            pred_label[fname] = 1

    # --- BƯỚC 3: TÍNH METRICS ---
    # Tạo ground truth cho tất cả các file đã quét
    gt_labels = get_ground_truth_robust(all_test_files)

    y_true = []
    y_pred = []

    print("True")
    cnt1, cnt0 = calc10(gt_labels)
    print("1: ", cnt1, " 0: ", cnt0)
    print("Predict")
    cnt1, cnt0 = calc10(pred_label)
    print("1: ", cnt1, " 0: ", cnt0)

    for fname in all_test_files:
        if fname in pred_label and fname in gt_labels:
            y_true.append(gt_labels[fname])
            y_pred.append(pred_label[fname])

    classifier_evaluation(y_true, y_pred)