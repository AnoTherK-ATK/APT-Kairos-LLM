#!/bin/bash

# Cấu hình thư mục
MODEL_DIR="./saved_models"
RESULT_DIR="./inference_results"
ARTIFACT_DIR="./artifact"

# Danh sách các model bạn đã train và copy sang
# Đảm bảo trong folder saved_models có các file: models_unimp.pt, models_sage.pt, ...
MODELS=("unimp" "sage" "gcn" "gat" "rgcn")

# Tạo thư mục lưu kết quả
mkdir -p $RESULT_DIR

echo "========================================================"
echo "   KAIROS - CHẾ ĐỘ CHẠY ĐÁNH GIÁ (INFERENCE ONLY)"
echo "========================================================"

# Kiểm tra xem có thư mục data chưa
if [ ! -f "$ARTIFACT_DIR/graphs/graph_4_6.TemporalData.simple" ]; then
    echo "❌ LỖI: Không tìm thấy dữ liệu pre-process (.simple files)."
    echo "👉 Hãy copy thư mục 'artifact' từ máy train sang máy này."
    exit 1
fi

for model in "${MODELS[@]}"
do
    rm ./artifact/evaluation.log
    # Kiểm tra xem file model có tồn tại không
    SOURCE_MODEL="$MODEL_DIR/models_$model.pt"

    if [ ! -f "$SOURCE_MODEL" ]; then
        echo "⚠️  Cảnh báo: Không tìm thấy $SOURCE_MODEL. Bỏ qua..."
        continue
    fi

    echo ""
    echo "--------------------------------------------------------"
    echo ">> Đang nạp mô hình: $model"
    echo "--------------------------------------------------------"

    # 1. Copy model về tên mặc định để test.py đọc
    # (Lý do: test.py mặc định load 'models.pt')
    cp "$SOURCE_MODEL" "$MODEL_DIR/models.pt"
    echo "   [+] Đã nạp trọng số (weights) thành công."

    # 2. Chạy Test (Tính Loss cho từng cạnh)
    # Bước này sẽ ghi đè các file .txt trong artifact/graph_4_x
    echo "   [+] Đang chạy Test (Reconstruction)..."
    python test.py > /dev/null 2>&1 # Ẩn bớt log rác nếu muốn

    # 3. Chạy Xây dựng hàng đợi bất thường
    # Bước này dùng kết quả của bước 2
    echo "   [+] Đang xây dựng hàng đợi bất thường..."
    python anomalous_queue_construction.py > /dev/null 2>&1

    # 4. Chạy Đánh giá
    echo "   [+] Đang tính toán chỉ số (Precision/Recall)..."
    python evaluation.py > temp_eval.log 2>&1

    # In kết quả tóm tắt ra màn hình ngay lập tức
    grep "F1-Score:" temp_eval.log
    grep "AUC:" temp_eval.log

    # 5. Lưu log kết quả
    cp "$ARTIFACT_DIR/evaluation.log" "$RESULT_DIR/evaluation_$model.log"
    echo "   [✓] Đã lưu kết quả vào: $RESULT_DIR/evaluation_$model.log"

done

echo ""
echo "========================================================"
echo "HOÀN TẤT! ĐANG VẼ BIỂU ĐỒ SO SÁNH..."
echo "========================================================"

# Gọi file python vẽ biểu đồ (dùng lại file plot_results.py ở câu trả lời trước)
if [ -f "plot_results.py" ]; then
    python plot_results.py
else
    echo "⚠️ Không tìm thấy file plot_results.py để vẽ biểu đồ."
fi