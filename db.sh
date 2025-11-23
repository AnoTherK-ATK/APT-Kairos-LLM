#!/bin/bash
set -e  # Dừng ngay nếu có lệnh bị lỗi

# 1. Cấu hình biến
DB_NAME=${1:-"tc_cadet_dataset_db"} # Lấy tham số 1 hoặc mặc định
DB_USER="postgres"
# Mật khẩu mặc định là 'postgres' nếu không được truyền vào
DB_PASS=${DB_PASSWORD:-"postgres"}

# 2. Kiểm tra quyền Root
if [ "$EUID" -ne 0 ]; then
  echo "❌ LỖI: Script này cần chạy với quyền root (sudo)."
  echo "👉 Hãy chạy: sudo ./db.sh"
  exit 1
fi

echo "=================================================="
echo "[*] Đang thiết lập Database: $DB_NAME"
echo "=================================================="

# Xuất biến môi trường mật khẩu để psql sử dụng (nếu cần)
export PGPASSWORD='$DB_PASS'

# 3. Tạo Database
echo "[*] Đang xóa (nếu có) và tạo lại database..."
sudo -u postgres psql <<EOF
DROP DATABASE IF EXISTS $DB_NAME;
CREATE DATABASE $DB_NAME;
EOF

# 4. Tạo Bảng (Tables)
echo "[*] Đang tạo cấu trúc bảng trong $DB_NAME..."
sudo -u postgres psql -d $DB_NAME <<EOF

-- Bảng lưu sự kiện (Edges)
CREATE TABLE IF NOT EXISTS event_table (
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    timestamp_rec bigint,
    _id           serial
);
ALTER TABLE event_table OWNER TO $DB_USER;
CREATE UNIQUE INDEX IF NOT EXISTS event_table__id_uindex ON event_table (_id);

-- Bảng File Node
CREATE TABLE IF NOT EXISTS file_node_table (
    node_uuid varchar NOT NULL,
    hash_id   varchar NOT NULL,
    path      varchar,
    CONSTRAINT file_node_table_pk PRIMARY KEY (node_uuid, hash_id)
);
ALTER TABLE file_node_table OWNER TO $DB_USER;

-- Bảng Netflow (Socket) Node
CREATE TABLE IF NOT EXISTS netflow_node_table (
    node_uuid varchar NOT NULL,
    hash_id   varchar NOT NULL,
    src_addr  varchar,
    src_port  varchar,
    dst_addr  varchar,
    dst_port  varchar,
    CONSTRAINT netflow_node_table_pk PRIMARY KEY (node_uuid, hash_id)
);
ALTER TABLE netflow_node_table OWNER TO $DB_USER;

-- Bảng Subject (Process) Node
CREATE TABLE IF NOT EXISTS subject_node_table (
    node_uuid varchar,
    hash_id   varchar,
    exec      varchar
);
ALTER TABLE subject_node_table OWNER TO $DB_USER;

-- Bảng ánh xạ Node ID
CREATE TABLE IF NOT EXISTS node2id (
    hash_id   varchar NOT NULL PRIMARY KEY,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
ALTER TABLE node2id OWNER TO $DB_USER;
CREATE UNIQUE INDEX IF NOT EXISTS node2id_hash_id_uindex ON node2id (hash_id);

EOF

echo "=================================================="
echo "[✓] HOÀN TẤT! Database đã sẵn sàng cho KAIROS."
echo "=================================================="