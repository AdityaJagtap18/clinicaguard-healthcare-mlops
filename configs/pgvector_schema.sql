-- PostgreSQL + pgvector Clinical Vector Store Schema
-- PRJ-03: ClinicaGuard

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS patient_notes CASCADE;

CREATE TABLE patient_notes (
    patient_id VARCHAR(50) PRIMARY KEY,
    admission_id VARCHAR(50) NOT NULL,
    admission_date TIMESTAMP NOT NULL,
    discharge_date TIMESTAMP NOT NULL,
    deidentified_note TEXT NOT NULL,
    note_embedding vector(384) NOT NULL,
    readmitted_30d INTEGER NOT NULL,
    cluster_id INTEGER
);

-- HNSW Cosine Distance Index for sub-millisecond semantic search
CREATE INDEX IF NOT EXISTS idx_patient_notes_hnsw 
ON patient_notes USING hnsw (note_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Standard B-Tree indexes for relational queries
CREATE INDEX IF NOT EXISTS idx_patient_admission ON patient_notes (admission_id);
CREATE INDEX IF NOT EXISTS idx_patient_readmit ON patient_notes (readmitted_30d);
