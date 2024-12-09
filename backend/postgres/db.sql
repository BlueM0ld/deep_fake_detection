-- Database initalisation 
-- Schema setup

CREATE TABLE IF NOT EXISTS user_logs (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    search_query TEXT NOT NULL,
    related_docs TEXT NOT NULL,
    selected_doc INT NOT NULL,
    -- unrelated_dodcs TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS documents (
    id INT PRIMARY KEY,
    document TEXT NOT NULL,
    document_encoding FLOAT8[] NOT NULL
);

CREATE TABLE IF NOT EXISTS videos (
    id SERIAL PRIMARY KEY,
    video_path TEXT NOT NULL,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id INT REFERENCES user_logs(id)  -- Assuming a relationship with user_logs
);