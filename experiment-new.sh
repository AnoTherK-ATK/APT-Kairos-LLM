#!/bin/bash

# Dọn dẹp các thư mục kết quả cũ
rm -rf ./saved_models/
rm -rf ./experiment_results/
mkdir -p ./saved_models/
mkdir -p ./experiment_results/

echo "=================================================="
echo "BƯỚC 1: KIỂM TRA VÀ THIẾT LẬP DATABASE"
echo "=================================================="

# Chạy setup DB và preprocess dữ liệu
make preprocess

if [ $? -ne 0 ]; then
    echo "❌ LỖI: Không thể thiết lập Database. Vui lòng kiểm tra PostgreSQL."
    exit 1
fi

echo ">> Database đã sẵn sàng."

# --- CHỈNH SỬA: CHỈ CHẠY UNIMP VÀ GAT ---
MODELS=("unimp" "gat")

echo "=================================================="
echo "BẮT ĐẦU CHẠY THỰC NGHIỆM: UNIMP & GAT"
echo "=================================================="

for model in "${MODELS[@]}"
do
    echo ""
    echo "--------------------------------------------------"
    echo ">> Đang chạy mô hình: $model"
    echo "--------------------------------------------------"

    # Chạy pipeline đầy đủ (bao gồm train, test, louvain, gnnexplainer)
    # Target này được định nghĩa trong Makefile cập nhật
    make pipeline_$model EPOCHS=50

    # Kiểm tra xem quá trình chạy có thành công không
    if [ $? -eq 0 ]; then
        echo "✅ Chạy thành công $model. Đang sao lưu kết quả..."

        # 1. Lưu Log Đánh giá (Evaluation Metrics)
        if [ -f "./artifact/evaluation.log" ]; then
            cp ./artifact/evaluation.log "./experiment_results/evaluation_$model.log"
            echo "   -> Đã lưu Evaluation Log"
        fi

        # 2. Lưu kết quả Attack Investigation (Subgraph từ Louvain)
        # File attack_investigation.py tạo ra folder artifact/graph_visual
        if [ -d "./artifact/graph_visual" ]; then
            cp -r ./artifact/graph_visual "./experiment_results/${model}_investigation_graphs"
            echo "   -> Đã lưu Investigation Graphs (Louvain)"
        else
            echo "   ⚠️ Cảnh báo: Không tìm thấy thư mục graph_visual"
        fi

        # 3. Lưu kết quả GNNExplainer (Critical Path)
        # File run_explainer_pipeline.py tạo ra artifact/critical_path_explained.dot
        if [ -f "./artifact/critical_path_explained.dot" ]; then
            # Copy file .dot gốc
            cp ./artifact/critical_path_explained.dot "./experiment_results/${model}_critical_path.dot"
            echo "   -> Đã lưu Critical Path (.dot)"

            # (Tuỳ chọn) Nếu có cài Graphviz, tự động render ra ảnh PNG
            if command -v dot &> /dev/null; then
                dot -Tpng "./artifact/critical_path_explained.dot" -o "./experiment_results/${model}_critical_path.png"
                echo "   -> Đã render Critical Path sang ảnh PNG"
            fi
        else
             echo "   ⚠️ Cảnh báo: Không tìm thấy file critical_path_explained.dot"
        fi

    else
        echo "❌ LỖI: Quá trình chạy model $model thất bại!"
    fi

    # Nghỉ 2 giây để hệ thống ổn định trước khi chạy model tiếp theo
    sleep 2
done

echo ""
echo "=================================================="
echo "HOÀN TẤT TOÀN BỘ THỰC NGHIỆM"
echo "Kết quả chi tiết nằm trong thư mục ./experiment_results/"
echo "=================================================="

# Vẽ biểu đồ so sánh (nếu có script vẽ)
if [ -f "plot_results.py" ]; then
    python plot_results.py
fi