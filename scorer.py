from rouge_score import rouge_scorer
import pandas as pd
import sys

# ---------------------------------------------------------
# 1. DỮ LIỆU GROUND TRUTH (Đã chuẩn hóa từ file PDF)
# ---------------------------------------------------------
cadets_ground_truth_en = [
    {
        "attack_id": "3.1",
        "timestamp": "2018-04-06 11:00:00",
        "description": "Exploited Nginx using a malformed HTTP request. Established a reverse connection to C2 server. Escalated privileges to root. Downloaded and executed a network reconnaissance tool (netrecon). Attempted to inject a malicious library into the sshd process (PID 809).",
        "iocs": "Attacker IPs: 81.49.200.166, 154.145.113.18, 61.167.39.128, 152.111.159.139, 139.123.0.113. Malicious Files: /var/log/devc, /tmp/vUgefal. Processes: nginx, sshd, netrecon.",
        "impact": "System crash (Kernel Panic) due to failed process injection."
    },
    {
        "attack_id": "3.8",
        "timestamp": "2018-04-11 15:00:00",
        "description": "Re-exploited Nginx service. Downloaded a malicious shared object file renamed as 'grain' to /tmp/. Attempted to inject this file into the sshd process (PID 802) for persistence.",
        "iocs": "Attacker IPs: 25.159.96.207, 155.162.39.48, 76.56.184.25, 198.115.236.119. Malicious Files: /tmp/grain, /tmp/sendmail. Processes: nginx, sshd.",
        "impact": "System crash (Kernel Panic) following the injection attempt."
    },
    {
        "attack_id": "3.13",
        "timestamp": "2018-04-12 14:00:00",
        "description": "Exploited Nginx. Dropped multiple implants to disk. Executed a root implant (XIM) and a user-level implant (Micro APT/sendmail). The user-level implant performed extensive port scanning of the internal network (128.55.12.0/24).",
        "iocs": "Attacker IPs: 25.159.96.207, 53.158.101.118, 98.15.44.232, 192.113.144.28. Malicious Files: tmux-1002, minions, font, XIM, netlog, sendmail, main, test. Processes: nginx, XIM, sendmail.",
        "impact": "Network reconnaissance (Port scan) and unauthorized execution."
    },
    {
        "attack_id": "3.14",
        "timestamp": "2018-04-13 09:00:00",
        "description": "Re-exploited Nginx. Downloaded an implant executable (pEja72mA) and a library (eWq10bVcx) to /tmp/. Executed the implant with root privileges. Attempted injection into sshd process (PID 20691) using a non-elevated injection module.",
        "iocs": "Attacker IPs: 25.159.96.207, 53.158.101.118, 198.115.236.119. Malicious Files: /tmp/pEja72mA, /tmp/eWq10bVcx, /tmp/memhelp.so, /tmp/eraseme, /tmp/done.so. Processes: nginx, sshd.",
        "impact": "Denial of Service (SSHD became unresponsive)."
    },
    {
        "attack_id": "4.1",
        "timestamp": "2018-04-06 15:00:00",
        "description": "Attacker connected to the Postfix email server on port 25 to send phishing emails impersonating internal users. This server acted as a relay for the phishing campaign.",
        "iocs": "Attacker IP: 62.83.155.175. Targeted Service: postfix (TCP/25).",
        "impact": "Unauthorized use of email server for phishing relay."
    }
]

cadets_ground_truth_vn = [
    {
        "attack_id": "3.1",
        "timestamp": "2018-04-06 11:00:00",
        "description": "Khai thác dịch vụ Nginx bằng một yêu cầu HTTP độc hại. Thiết lập kết nối ngược (reverse connection) về máy chủ C2. Leo thang đặc quyền lên root. Tải xuống và thực thi công cụ thám thính mạng (netrecon). Cố gắng tiêm thư viện độc hại vào tiến trình sshd (PID 809).",
        "iocs": "IP Kẻ tấn công: 81.49.200.166, 154.145.113.18, 61.167.39.128, 152.111.159.139, 139.123.0.113. Tệp độc hại: /var/log/devc, /tmp/vUgefal. Tiến trình: nginx, sshd, netrecon.",
        "impact": "Sập hệ thống (Kernel Panic) do quá trình tiêm mã (injection) thất bại."
    },
    {
        "attack_id": "3.8",
        "timestamp": "2018-04-11 15:00:00",
        "description": "Tái khai thác dịch vụ Nginx. Tải xuống tệp shared object độc hại và đổi tên thành 'grain' tại thư mục /tmp/. Cố gắng tiêm tệp này vào tiến trình sshd (PID 802) để duy trì quyền truy cập.",
        "iocs": "IP Kẻ tấn công: 25.159.96.207, 155.162.39.48, 76.56.184.25, 198.115.236.119. Tệp độc hại: /tmp/grain, /tmp/sendmail. Tiến trình: nginx, sshd.",
        "impact": "Sập hệ thống (Kernel Panic) ngay sau nỗ lực tiêm mã."
    },
    {
        "attack_id": "3.13",
        "timestamp": "2018-04-12 14:00:00",
        "description": "Khai thác Nginx. Thả nhiều mã độc xuống đĩa cứng. Thực thi mã độc quyền root (XIM) và mã độc quyền người dùng thường (Micro APT/sendmail). Mã độc quyền người dùng thực hiện quét cổng (port scanning) mạng nội bộ (dải 128.55.12.0/24).",
        "iocs": "IP Kẻ tấn công: 25.159.96.207, 53.158.101.118, 98.15.44.232, 192.113.144.28. Tệp độc hại: tmux-1002, minions, font, XIM, netlog, sendmail, main, test. Tiến trình: nginx, XIM, sendmail.",
        "impact": "Thám thính mạng (Reconnaissance) và thực thi mã trái phép."
    },
    {
        "attack_id": "3.14",
        "timestamp": "2018-04-13 09:00:00",
        "description": "Tái khai thác Nginx. Tải xuống một tệp thực thi mã độc (pEja72mA) và thư viện (eWq10bVcx) vào /tmp/. Thực thi mã độc với quyền root. Cố gắng tiêm vào tiến trình sshd (PID 20691) sử dụng module tiêm không nâng quyền.",
        "iocs": "IP Kẻ tấn công: 25.159.96.207, 53.158.101.118, 198.115.236.119. Tệp độc hại: /tmp/pEja72mA, /tmp/eWq10bVcx, /tmp/memhelp.so, /tmp/eraseme, /tmp/done.so. Tiến trình: nginx, sshd.",
        "impact": "Từ chối dịch vụ (Tiến trình SSHD bị treo/không phản hồi)."
    },
    {
        "attack_id": "4.1",
        "timestamp": "2018-04-06 15:00:00",
        "description": "Kẻ tấn công kết nối đến máy chủ email Postfix trên cổng 25 để gửi email lừa đảo (phishing) mạo danh người dùng nội bộ. Máy chủ này đóng vai trò trung chuyển (relay) cho chiến dịch lừa đảo.",
        "iocs": "IP Kẻ tấn công: 62.83.155.175. Dịch vụ bị lạm dụng: postfix (TCP/25).",
        "impact": "Sử dụng trái phép máy chủ email để phát tán thư rác/lừa đảo."
    }
]

# Hàm tạo văn bản mẫu (Reference Text) bằng tiếng Việt cho ROUGE
def create_reference_text_vn(event):
    text = (
        f"Kịch bản tấn công {event['attack_id']} được phát hiện lúc {event['timestamp']}. "
        f"{event['description']} "
        f"Các dấu hiệu xâm nhập (IOCs) bao gồm {event['iocs']} "
        f"Hậu quả chính là {event['impact']}."
    )
    return text

if(sys.argv[1] == "en"):
    cadets_ground_truth = cadets_ground_truth_en
else:
    cadets_ground_truth = cadets_ground_truth_vn

# ---------------------------------------------------------
# 2. HÀM CHUYỂN ĐỔI GROUND TRUTH THÀNH VĂN BẢN (REFERENCE)
# ---------------------------------------------------------
def create_reference_text_en(event):
    text = (
        f"Attack {event['attack_id']} detected at {event['timestamp']}. "
        f"{event['description']} "
        f"Indicators include {event['iocs']} "
        f"The impact was {event['impact']}."
    )
    return text

# ---------------------------------------------------------
# 3. GIẢ LẬP OUTPUT CỦA LLM (HYPOTHESIS)
# ---------------------------------------------------------
# Đây là ví dụ báo cáo mà mô hình của bạn có thể sinh ra
llm_generated_outputs = ""

# ---------------------------------------------------------
# 4. TÍNH TOÁN ROUGE SCORE
# ---------------------------------------------------------
def calculate_metrics(ground_truth_list, llm_outputs):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = []

    for event in ground_truth_list:
        attack_id = event['attack_id']
        
        # 1. Tạo Reference (Văn bản chuẩn)
        if(sys.argv[1] == "en"):
            reference_text = create_reference_text_en(event)
        else:
            reference_text = create_reference_text_vn(event)
        
        # 2. Lấy Hypothesis (Văn bản do LLM sinh ra)
        hypothesis_text = llm_outputs

        # 3. Tính điểm
        scores = scorer.score(reference_text, hypothesis_text)
        
        results.append({
            "Attack ID": attack_id,
            "ROUGE-1 (F1)": round(scores['rouge1'].fmeasure, 4),
            "ROUGE-2 (F1)": round(scores['rouge2'].fmeasure, 4),
            "ROUGE-L (F1)": round(scores['rougeL'].fmeasure, 4),
            "Reference": reference_text,
            "Generated": hypothesis_text
        })

    return pd.DataFrame(results)

# ---------------------------------------------------------
# 5. CHẠY VÀ HIỂN THỊ KẾT QUẢ
# ---------------------------------------------------------
with open(sys.argv[2], "r", encoding="utf-8") as f:
    llm_generated_outputs = f.read()
df_results = calculate_metrics(cadets_ground_truth, llm_generated_outputs)

# Hiển thị bảng kết quả
print(df_results[["Attack ID", "ROUGE-1 (F1)", "ROUGE-2 (F1)", "ROUGE-L (F1)"]])

# Tính trung bình toàn bộ tập test
print("\n--- AVERAGE SCORES ---")
print(df_results[["ROUGE-1 (F1)", "ROUGE-2 (F1)", "ROUGE-L (F1)"]].mean())