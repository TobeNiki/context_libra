# PostgreSQL 17 + PGroonga (Alpine) ベースイメージ
FROM groonga/pgroonga:4.0.4-alpine-17

ENV LANG=ja_JP.utf8
ENV LC_ALL=ja_JP.utf8

USER root

# pgvector をソースからビルド
RUN apk update && \
    apk add --no-cache --virtual .build-deps \
      build-base \
      git \
      postgresql-dev \
      clang19 \
      llvm19 \
    && git clone --depth 1 https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && cd /tmp/pgvector \
    # 念のため pg_config を明示
    && make PG_CONFIG=/usr/local/bin/pg_config \
    && make PG_CONFIG=/usr/local/bin/pg_config install \
    && cd / \
    && rm -rf /tmp/pgvector \
    && apk del .build-deps \
    && rm -rf /var/cache/apk/*

# postgres ユーザーに戻す
USER postgres

# 初期起動時に拡張を自動有効化する SQL
COPY init-extensions.sql /docker-entrypoint-initdb.d/init-extensions.sql
