DROP DATABASE IF EXISTS tc_cadet_dataset_db;
CREATE DATABASE tc_cadet_dataset_db;

CREATE TABLE IF NOT EXISTS event_table (
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    timestamp_rec bigint,
    _id           serial
);
ALTER TABLE event_table OWNER TO postgres;
CREATE UNIQUE INDEX IF NOT EXISTS event_table__id_uindex ON event_table (_id);

CREATE TABLE IF NOT EXISTS file_node_table (
    node_uuid varchar NOT NULL,
    hash_id   varchar NOT NULL,
    path      varchar,
    CONSTRAINT file_node_table_pk PRIMARY KEY (node_uuid, hash_id)
);
ALTER TABLE file_node_table OWNER TO postgres;

CREATE TABLE IF NOT EXISTS netflow_node_table (
    node_uuid varchar NOT NULL,
    hash_id   varchar NOT NULL,
    src_addr  varchar,
    src_port  varchar,
    dst_addr  varchar,
    dst_port  varchar,
    CONSTRAINT netflow_node_table_pk PRIMARY KEY (node_uuid, hash_id)
);
ALTER TABLE netflow_node_table OWNER TO postgres;

CREATE TABLE IF NOT EXISTS subject_node_table (
    node_uuid varchar,
    hash_id   varchar,
    exec      varchar
);
ALTER TABLE subject_node_table OWNER TO postgres;

CREATE TABLE IF NOT EXISTS node2id (
    hash_id   varchar NOT NULL PRIMARY KEY,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
ALTER TABLE node2id OWNER TO postgres;
CREATE UNIQUE INDEX IF NOT EXISTS node2id_hash_id_uindex ON node2id (hash_id);