#!/bin/bash
rm -rf ./saved_models/
rm -rf ./experiment_results/
mkdir -p ./saved_models/
mkdir -p ./experiment_results/
echo "=================================================="
echo "BƯỚC 1: KIỂM TRA VÀ THIẾT LẬP DATABASE"
echo "=================================================="

# Chạy setup DB. Lệnh này sẽ hỏi password sudo nếu cần.
make preprocess

if [ $? -ne 0 ]; then
    echo "❌ LỖI: Không thể thiết lập Database. Vui lòng kiểm tra PostgreSQL."
    exit 1
fi

echo ">> Database đã sẵn sàng."

# Danh sách các mô hình muốn chạy thử nghiệm
# Đảm bảo tên này khớp với tên target trong Makefile (ví dụ: pipeline_unimp, pipeline_sage)
# Nếu bạn chưa cập nhật Makefile cho gcn/rgcn thì bỏ chúng ra khỏi list dưới đây
MODELS=("sage" "gcn" "rgcn")

echo "=================================================="
echo "BẮT ĐẦU CHẠY THỰC NGHIỆM TỰ ĐỘNG KAIROS"
echo "=================================================="

for model in "${MODELS[@]}"
do
    echo ""
    echo "--------------------------------------------------"
    echo ">> Đang chạy mô hình: $model"
    echo "--------------------------------------------------"

    # 1. Dọn dẹp artifact cũ để đảm bảo không bị lẫn kết quả
    # (Tuỳ chọn: nếu Makefile của bạn đã handle việc này thì có thể bỏ qua)
    # rm -rf ./artifact/graph_* # 2. Gọi lệnh Make tương ứng
    # Giả định Makefile của bạn đã có các target: pipeline_unimp, pipeline_sage...
    # EPOCHS=50 là tham số truyền vào để train nhanh hoặc chậm
    make pipeline_$model EPOCHS=50

    # 3. Kiểm tra xem chạy có thành công không
    if [ $? -eq 0 ]; then
        echo ">> Chạy thành công $model. Đang lưu log..."

        # 4. Copy file log ra thư mục kết quả và đổi tên
        # Ví dụ: experiment_results/evaluation_unimp.log
        cp ./artifact/evaluation.log ./experiment_results/evaluation_$model.log

        echo ">> Đã lưu log tại: ./experiment_results/evaluation_$model.log"
    else
        echo ">> LỖI: Quá trình chạy model $model thất bại!"
    fi

    # Nghỉ 2 giây trước khi chạy cái tiếp theo
    sleep 2
done

echo ""
echo "=================================================="
echo "HOÀN TẤT TOÀN BỘ THỰC NGHIỆM"
echo "Kết quả nằm trong thư mục ./experiment_results"
echo "Chạy file python plot_results.py để vẽ biểu đồ."
echo "=================================================="

python plot_results.py

