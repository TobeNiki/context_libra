# PostgreSQL 17 + PGroonga (Alpine) ベースイメージ
# FROM groonga/pgroonga:4.0.4-alpine-17
FROM postgres:17-alpine

ENV PGROONGA_VERSION=4.0.4 \
    GROONGA_VERSION=15.1.7 \
    LANG=ja_JP.utf8 \
    LC_ALL=ja_JP.utf8
# ビルドツールのインストールと PGroonga のソースからのビルド
COPY build.sh /
RUN chmod +x /build.sh && \
  apk add --no-cache --virtual .build-deps \
    apache-arrow-dev \
    build-base \
    clang19-dev \
    cmake \
    gettext-dev \
    linux-headers \
    llvm19 \
    lz4-dev \
    msgpack-c-dev \
    postgresql-dev \
    rapidjson-dev \
    ruby \
    samurai \
    wget \
    xsimd-dev \
    xxhash-dev \
    zlib-dev \
    zstd-dev && \
  /build.sh ${PGROONGA_VERSION} ${GROONGA_VERSION}

RUN rm -f build.sh && \
    apk del .build-deps
# ランタイム依存関係のインストール
RUN apk add --no-cache \
    libarrow \
    libgomp \
    libxxhash \
    msgpack-c \
    zlib \
    zstd 

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
