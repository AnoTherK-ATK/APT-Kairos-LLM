#!/bin/bash

# --- CẤU HÌNH CÁC BIẾN (Bạn hãy sửa lại đường dẫn tại đây) ---
SOURCE_DIR="./artifact"            # Thư mục chứa file .dot
DEST_DIR="./graph-output"         # Thư mục muốn copy đến
CONDA_ENV_NAME="kairos312"    # Tên môi trường Conda của bạn
PYTHON_SCRIPT="llm_analyzer.py"    # Đường dẫn file python

# 1. Tạo thư mục đích nếu chưa tồn tại
if [ ! -d "$DEST_DIR" ]; then
  echo "Thư mục $DEST_DIR chưa tồn tại. Đang tạo mới..."
  mkdir -p "$DEST_DIR"
fi

conda activate "$CONDA_ENV_NAME"
python "run_explainer_pipeline.py"


# 2. Copy file .dot từ artifact sang thư mục đích
echo "Đang copy các file .dot từ $SOURCE_DIR sang $DEST_DIR..."
# Kiểm tra xem có file .dot không để tránh lỗi nếu thư mục rỗng
if ls "$SOURCE_DIR"/*.dot 1> /dev/null 2>&1; then
    cp "$SOURCE_DIR"/*.dot "$DEST_DIR/"
    echo "Copy hoàn tất."
else
    echo "Cảnh báo: Không tìm thấy file .dot nào trong $SOURCE_DIR."
fi

# 3. Chạy code Python trong môi trường Conda
echo "Đang chuẩn bị chạy $PYTHON_SCRIPT trong môi trường $CONDA_ENV_NAME..."

# Cách 1: Sử dụng 'conda run' (Khuyên dùng cho script vì ổn định hơn)
# Lệnh này chạy python trong môi trường ảo mà không cần activate toàn bộ shell
eval "$(conda shell.bash hook)"
cp "$PYTHON_SCRIPT" "$DEST_DIR"
conda activate "$CONDA_ENV_NAME"
python "$DEST_DIR/$PYTHON_SCRIPT"

# Nếu cách trên gặp lỗi, bạn có thể dùng lệnh ngắn gọn hơn dưới đây (bỏ comment):
# conda run -n "$CONDA_ENV_NAME" python "$PYTHON_SCRIPT"

echo "Đã thực hiện xong."