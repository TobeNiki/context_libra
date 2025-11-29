#!/bin/sh
set -eux

PGROONGA_VERSION=$1
GROONGA_VERSION=$2

# Build and install MeCab
cd /tmp
wget "https://github.com/shogo82148/mecab/releases/download/v0.996.12/mecab-0.996.12.tar.gz"
tar xf mecab-0.996.12.tar.gz
cd mecab-0.996.12
./configure --prefix=/usr/local --with-charset=utf8
make
make install
cd /tmp
rm -rf mecab-0.996.12 mecab-0.996.12.tar.gz

# Build and install MeCab IPA dictionary
cd /tmp
wget "https://github.com/shogo82148/mecab/releases/download/v0.996.12/mecab-ipadic-2.7.0-20070801.tar.gz"
tar xf mecab-ipadic-2.7.0-20070801.tar.gz
cd mecab-ipadic-2.7.0-20070801
./configure --prefix=/usr/local --with-charset=utf8 --with-mecab-config=/usr/local/bin/mecab-config
make
make install
cd /tmp
rm -rf mecab-ipadic-2.7.0-20070801 mecab-ipadic-2.7.0-20070801.tar.gz

# Build and install Groonga with MeCab support
cd /tmp
wget "https://packages.groonga.org/source/groonga/groonga-${GROONGA_VERSION}.tar.gz"
tar xf "groonga-${GROONGA_VERSION}.tar.gz"
cd "groonga-${GROONGA_VERSION}"
cmake \
  -S . \
  -B ../groonga.build \
  --preset=release-maximum \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DGRN_WITH_MRUBY=OFF \
  -DGRN_WITH_MECAB=ON
cmake --build ../groonga.build
cmake --install ../groonga.build
cd /tmp
rm -rf "groonga-${GROONGA_VERSION}" "groonga-${GROONGA_VERSION}.tar.gz" groonga.build

# Build and install PGroonga
cd /tmp
wget "https://packages.groonga.org/source/pgroonga/pgroonga-${PGROONGA_VERSION}.tar.gz"
tar xf "pgroonga-${PGROONGA_VERSION}.tar.gz"
cd "pgroonga-${PGROONGA_VERSION}"
make HAVE_MSGPACK=1 HAVE_XXHASH=1 \
  PG_CPPFLAGS="-I/usr/local/include/groonga -DPGRN_VERSION='\"${PGROONGA_VERSION}\"'" \
  SHLIB_LINK="-L/usr/local/lib -lgroonga"
make install
cd /tmp
rm -rf "pgroonga-${PGROONGA_VERSION}" "pgroonga-${PGROONGA_VERSION}.tar.gz"