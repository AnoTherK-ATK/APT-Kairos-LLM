Chào bạn, tôi là chuyên gia điều tra số và phân tích mã độc. Dựa trên dữ liệu "Provenance Graph" từ hệ thống KAIROS mà bạn cung cấp, tôi đã thực hiện phân tích sâu chuỗi các sự kiện, đối chiếu các điểm bất thường (Anomaly Score) và các chỉ số IOC (Indicator of Compromise).

Dưới đây là báo cáo điều tra chi tiết.

---

# BÁO CÁO ĐIỀU TRA SỐ & PHÂN TÍCH TẤN CÔNG APT

**Ngày phân tích:** 24/05/2024
**Nguồn dữ liệu:** KAIROS Anomaly Detection System
**Đối tượng:** Hệ thống máy chủ (Dấu hiệu cho thấy đây là hệ điều hành **FreeBSD** dựa trên các đường dẫn như `/var/db/entropy`, `/etc/pwd.db`, `kenv`, `/usr/libexec/save-entropy`).

## 1. TỔNG QUAN CÁC DẤU HIỆU BẤT THƯỜNG (KEY ANOMALIES)

Dựa trên điểm số bất thường (Loss Score), tôi đã khoanh vùng các thực thể và hành vi nguy hiểm nhất:

### A. Hành vi thực thi và sửa đổi hệ thống (Critical Execution & Persistence)
*   **`sh` -> `/usr/libexec/save-entropy` (Loss: 8.005)**: Đây là điểm bất thường cao nhất. `save-entropy` là tác vụ hệ thống (thường chạy qua cron hoặc shutdown). Việc `sh` gọi trực tiếp tiến trình này một cách bất thường gợi ý kẻ tấn công đang lợi dụng nó để **duy trì sự hiện diện (Persistence)** hoặc che giấu mã độc trong quá trình khởi động/tắt máy.
*   **`sh` -> `/sbin/resolvconf` (Loss: 7.989) & `/sbin/dhclient-script` (Loss: 6.617)**: Kẻ tấn công đang can thiệp vào cấu hình mạng và DNS. Điều này thường nhằm mục đích **DNS Hijacking** hoặc đảm bảo kết nối C2 (Command & Control) không bị chặn.
*   **Tiến trình lạ `vUgefal`**: Một tiến trình có tên ngẫu nhiên, thực hiện ghi vào `/dev/null` (che giấu đầu ra) và kết nối mạng. Đây chắc chắn là **Malware Dropper** hoặc **Backdoor**.

### B. Xâm nhập ban đầu và Tải xuống (Initial Access & Payload)
*   **`nginx` & `php-fpm`**: Có dấu hiệu bị khai thác (Loss ~5.0). `nginx` giao tiếp với IP lạ `200.36.109.214` và ghi log bất thường.
*   **`wget` -> `/usr/home/user/eraseme/www.a7.org/index.html` (Loss: 4.948)**: Lệnh `wget` được sử dụng để tải file từ IP `69.20.49.234`. Đường dẫn chứa từ khóa `eraseme` (xóa tôi đi) và cấu trúc thư mục web gợi ý việc tải xuống **Phishing kit** hoặc **Defacement tool**.

### C. Do thám và Leo thang đặc quyền (Recon & PrivEsc)
*   **`lsof` -> `/dev/kmem` (Loss: 3.986)**: Việc mở thiết bị bộ nhớ kernel (`/dev/kmem`) là cực kỳ nguy hiểm. Các công cụ thông thường hiếm khi làm việc này trực tiếp trừ khi đó là hành vi của **Rootkit** đang cố gắng thao túng kernel hoặc đọc mật khẩu từ bộ nhớ.
*   **`sudo` execution**: Có nỗ lực sử dụng quyền root.
*   **Các lệnh do thám**: `uname`, `ps`, `netstat`, `kenv` được thực thi với điểm bất thường đồng nhất (~5.89), cho thấy một script tự động đang thu thập thông tin hệ thống.

### D. Hành vi Spam/Phishing nội bộ (Action on Objectives)
*   **Ghi file Mail hàng loạt**: `local`, `imapd`, `alpine` ghi dữ liệu vào `/var/mail/` của hàng loạt user (user, frank, bob, admin, alice, root...) với điểm Loss rất cao (>6.0).
*   **Kết nối SMTP**: Tiến trình `smtpd` kết nối đến nhiều cổng lạ trên IP `128.55.12.166`.
*   => **Kết luận:** Máy chủ đã bị biến thành một **Spambot** hoặc đang lây lan mã độc qua email nội bộ.

---

## 2. TÁI HIỆN KỊCH BẢN TẤN CÔNG (ATTACK SCENARIO)

Dựa trên các manh mối trên, tôi dựng lại kịch bản tấn công theo trình tự thời gian:

### Giai đoạn 1: Xâm nhập (Initial Compromise) - Ngày 06/04
1.  Kẻ tấn công khai thác lỗ hổng trên ứng dụng Web chạy **Nginx/PHP-FPM** (có thể là lỗi RCE hoặc SQLi).
2.  Tiến trình `nginx` hoặc `php-fpm` bị ép thực thi lệnh shell.
3.  Một mã độc tên là **`vUgefal`** được kích hoạt. Nó kết nối đến C2 server (`139.123.0.113`) để nhận lệnh.

### Giai đoạn 2: Thiết lập chỗ đứng & Leo thang (Persistence & Privilege Escalation) - Ngày 12/04
1.  Kẻ tấn công quay lại, sử dụng `wget` tải công cụ/payload từ `69.20.49.234`.
2.  Thực hiện do thám bằng `netstat`, `ps`, `uname`.
3.  **Leo thang đặc quyền:** Sử dụng `sudo` và đáng ngại hơn là `lsof` truy cập `/dev/kmem` (có thể đang cài cắm Rootkit cấp kernel).
4.  **Duy trì quyền truy cập:** Can thiệp vào các script hệ thống quan trọng của FreeBSD như `/sbin/resolvconf` và `/usr/libexec/save-entropy` (thông qua `sh`). Đây là kỹ thuật tinh vi để mã độc tự khởi chạy lại hoặc điều hướng traffic mạng.

### Giai đoạn 3: Thực hiện hành vi độc hại (Impact/Spamming) - Ngày 13/04
1.  Hệ thống thư điện tử (`smtpd`, `imapd`, `local`) bị chiếm quyền điều khiển.
2.  Tiến trình `alpine` (Email client dòng lệnh) được sử dụng tự động (script hóa) để gửi/nhận mail.
3.  Kẻ tấn công thực hiện chiến dịch **Mass Mailing/Spam**: Ghi đè vào hộp thư của toàn bộ user trên hệ thống (`/var/mail/*`) và gửi mail ra ngoài liên tục.

---

## 3. SƠ ĐỒ HÀNH VI TẤN CÔNG (MERMAID)

```mermaid
graph TD
    subgraph "External Threats (C2 & Dropzones)"
        IP_Exploit[200.36.109.214:80] -->|Exploit| Nginx
        IP_Drop[69.20.49.234:80] -->|Download Payload| Wget
        IP_C2[139.123.0.113:80] <-->|Command & Control| Malware_vUgefal
    end

    subgraph "Compromised Host (FreeBSD)"
        Nginx[Nginx / PHP-FPM] -->|Spawn| Sh[Shell / Bash]
        
        Sh -->|Exec| Wget
        Sh -->|Exec| Malware_vUgefal[Malware: vUgefal]
        
        subgraph "Privilege Escalation & Recon"
            Sh -->|Exec| Recon[uname, ps, netstat]
            Sh -->|Exec| Sudo
            Sh -->|Exec| Lsof
            Lsof -.->|Read Kernel Memory!| DevKmem[/dev/kmem]
        end

        subgraph "Persistence Mechanisms (Critical Anomalies)"
            Sh -->|Hijack/Modify| SaveEntropy[/usr/libexec/save-entropy]
            Sh -->|Hijack/Modify| ResolvConf[/sbin/resolvconf]
            Sh -->|Modify| Dhclient[/sbin/dhclient-script]
        end

        subgraph "Action on Objectives: SPAMBOT"
            Sh -->|Control| Alpine[Alpine Mail Client]
            Sh -->|Control| Local[Postfix Local]
            Local -->|Mass Write| Mailbox[/var/mail/user, root, admin...]
            Alpine -->|Write| DebugFiles[/home/user/.pine-debug*]
            Smtpd[SMTPD] -->|Outbound Spam| NetInternal[128.55.12.166:Multiple_Ports]
        end
    end

    style Malware_vUgefal fill:#ff6666,stroke:#333,stroke-width:2px
    style SaveEntropy fill:#ff0000,stroke:#333,stroke-width:4px
    style DevKmem fill:#ff0000,stroke:#333,stroke-width:4px
```

---

## 4. KẾT LUẬN VÀ KHUYẾN NGHỊ

### Kết luận
Đây là một cuộc tấn công đã thành công vào máy chủ FreeBSD. Kẻ tấn công đã chuyển từ việc xâm nhập qua Web sang kiểm soát hoàn toàn hệ thống (Root level access có khả năng cao do can thiệp vào `/dev/kmem` và `/usr/libexec`).
Mục đích cuối cùng của cuộc tấn công này (tại thời điểm ghi nhận) là **biến máy chủ thành một trạm trung chuyển Spam (Spambot)** và thiết lập cơ chế ẩn mình sâu trong hệ thống.

### Khuyến nghị xử lý
1.  **Cô lập ngay lập tức:** Ngắt kết nối mạng của máy chủ để ngăn chặn việc gửi spam và kết nối C2.
2.  **Thu thập chứng cứ:** Sao chụp (dump) RAM để phân tích tiến trình `vUgefal` và kiểm tra xem kernel có bị sửa đổi không (do dấu hiệu `/dev/kmem`).
3.  **Kiểm tra tính toàn vẹn:** Đối chiếu checksum của các file hệ thống quan trọng: `/usr/libexec/save-entropy`, `/sbin/resolvconf`, `/sbin/dhclient-script`. Khả năng cao các file này đã bị chèn mã độc.
4.  **Rà soát tài khoản:** Các tài khoản user (`frank`, `bob`, `alice`...) có thể đã bị thỏa hiệp mật khẩu hoặc bị dùng để phát tán mail nội bộ.
5.  **Dựng lại hệ thống:** Do có dấu hiệu can thiệp sâu vào hệ thống (Persistence) và Kernel, việc cài đặt lại hệ điều hành sạch (Re-image) được khuyến nghị hơn là cố gắng gỡ bỏ mã độc.