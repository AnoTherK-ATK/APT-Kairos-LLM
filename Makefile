EPOCHS ?= 50

DB_PASSWORD ?= postgres

prepare:
	rm -rf ./artifact/
	mkdir -p ./artifact/



# --- DATABASE SETUP ---
# Target này chạy script db.sh với quyền sudo và truyền biến môi trường
setup_db:
	@echo ">> Yêu cầu quyền Root để thiết lập PostgreSQL..."
	sudo DB_PASSWORD=$(DB_PASSWORD) ./db.sh

import_data:
	python create_database.py

create_database: setup_db import_data

embed_graphs:
	python embedding.py

# --- TRAINING TARGETS ---

# Train UniMP (GAT-based, Original KAIROS)
train_unimp:
	python train.py --model unimp --epoch $(EPOCHS)
	# Copy model to default path so test.py can pick it up
	cp ./artifact/models/models.pt ./saved_models/models_unimp.pt

# Train GraphSAGE
train_sage:
	python train.py --model sage --epoch $(EPOCHS)
	cp ./artifact/models/models.pt ./saved_models/models_sage.pt

# Train GCN
train_gcn:
	python train.py --model gcn --epoch $(EPOCHS)
	cp ./artifact/models/models.pt ./saved_models/models_gcn.pt

# Train RGCN
train_rgcn:
	python train.py --model rgcn --epoch $(EPOCHS)
	cp ./artifact/models/models.pt ./saved_models/models_rgcn.pt

# Train GAT
train_gat:
	python train.py --model gat --epoch $(EPOCHS)
	cp ./artifact/models/models.pt ./saved_models/models_gat.pt

test:
	python test.py

anomalous_queue:
	python anomalous_queue_construction.py

evaluation:
	python evaluation.py

attack_investigation:
	python attack_investigation.py


# Helper group
anomaly_detection: anomalous_queue evaluation

preprocess: prepare create_database embed_graphs

# =============================================================
pipeline: preprocess train_unimp test anomaly_detection attack_investigation

# Pipeline chuẩn (UniMP)
pipeline_unimp: prepare embed_graphs train_unimp test anomaly_detection

# Pipeline cho GraphSAGE
pipeline_sage: prepare embed_graphs train_sage test anomaly_detection

# Pipeline cho GCN
pipeline_gcn: prepare embed_graphs train_gcn test anomaly_detection

# Pipeline cho RGCN
pipeline_rgcn: prepare embed_graphs train_rgcn test anomaly_detection

# Pipeline cho GAT
pipeline_gat: prepare embed_graphs train_gat test anomaly_detection

