-- init-extensions.sql
-- 初回 initdb 時に一度だけ実行される (公式 Postgres イメージと同じ仕組み)

-- デフォルト DB "postgres" に拡張を有効化
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgroonga;