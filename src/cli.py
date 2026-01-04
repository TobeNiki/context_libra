#!/usr/bin/env python
"""
MCP RAG Server CLI

インデックスのクリアとインデックス化を行うためのコマンドラインインターフェース
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

from src.rag.character.character_rag_service import create_rag_service_from_env


def setup_logging():
    """
    ロギングの設定
    """
    # ログディレクトリの作成
    os.makedirs("logs", exist_ok=True)

    # ロギングの設定
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join("logs", "mcp_rag_cli.log"), encoding="utf-8"),
        ],
    )
    return logging.getLogger("cli")


def clear_index():
    """
    インデックスをクリアする
    """
    logger = setup_logging()
    logger.info("インデックスをクリアしています...")

    # 環境変数の読み込み
    load_dotenv()

    # RAGサービスの作成
    rag_service = create_rag_service_from_env()

    # 処理済みディレクトリのパス
    processed_dir = os.environ.get("PROCESSED_DIR", "data/processed")

    # ファイルレジストリの削除
    registry_path = Path(processed_dir) / "file_registry.json"
    if registry_path.exists():
        try:
            registry_path.unlink()
            logger.info(f"ファイルレジストリを削除しました: {registry_path}")
            print(f"ファイルレジストリを削除しました: {registry_path}")
        except Exception as e:
            logger.error(f"ファイルレジストリの削除に失敗しました: {str(e)}")
            print(f"ファイルレジストリの削除に失敗しました: {str(e)}")

    # インデックスをクリア
    result = rag_service.clear_index()

    if result["success"]:
        logger.info(f"インデックスをクリアしました（{result['deleted_count']} ドキュメントを削除）")
        print(f"インデックスをクリアしました（{result['deleted_count']} ドキュメントを削除）")
    else:
        logger.error(f"インデックスのクリアに失敗しました: {result.get('error', '不明なエラー')}")
        print(f"インデックスのクリアに失敗しました: {result.get('error', '不明なエラー')}")
        sys.exit(1)


def index_documents(directory_path, chunk_size=500, chunk_overlap=100, incremental=False):
    """
    ドキュメントをインデックス化する

    Args:
        directory_path: インデックス化するドキュメントが含まれるディレクトリのパス
        chunk_size: チャンクサイズ（文字数）
        chunk_overlap: チャンク間のオーバーラップ（文字数）
        incremental: 差分のみをインデックス化するかどうか
    """
    logger = setup_logging()
    if incremental:
        logger.info(f"ディレクトリ '{directory_path}' 内の差分ファイルをインデックス化しています...")
    else:
        logger.info(f"ディレクトリ '{directory_path}' 内のドキュメントをインデックス化しています...")

    # 環境変数の読み込み
    load_dotenv()

    # ディレクトリの存在確認
    if not os.path.exists(directory_path):
        logger.error(f"ディレクトリ '{directory_path}' が見つかりません")
        print(f"エラー: ディレクトリ '{directory_path}' が見つかりません")
        sys.exit(1)

    if not os.path.isdir(directory_path):
        logger.error(f"'{directory_path}' はディレクトリではありません")
        print(f"エラー: '{directory_path}' はディレクトリではありません")
        sys.exit(1)

    # RAGサービスの作成
    rag_service = create_rag_service_from_env()

    # 処理済みディレクトリのパス
    processed_dir = os.environ.get("PROCESSED_DIR", "data/processed")

    # インデックス化を実行
    if incremental:
        print(f"ディレクトリ '{directory_path}' 内の差分ファイルをインデックス化しています...")
    else:
        print(f"ディレクトリ '{directory_path}' 内のドキュメントをインデックス化しています...")

    # 進捗状況を表示するためのカウンタ
    processed_files = 0

    # 処理前にファイル数を取得
    total_files = 0
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file_path)[1].lower()
            if ext in [".md", ".markdown", ".txt", ".pdf", ".ppt", ".pptx", ".doc", ".docx"]:
                total_files += 1

    print(f"合計 {total_files} 個のファイルを検索しました...")

    # 元のRAGServiceのindex_documentsメソッドを呼び出す前に、
    # DocumentProcessorのprocess_directoryメソッドをオーバーライドして進捗を表示
    original_process_directory = rag_service.document_processor.process_directory

    def process_directory_with_progress(source_dir, processed_dir, chunk_size=500, overlap=100, incremental=False):
        nonlocal processed_files
        results = []
        source_directory = Path(source_dir)

        if not source_directory.exists() or not source_directory.is_dir():
            logger.error(f"ディレクトリ '{source_dir}' が見つからないか、ディレクトリではありません")
            raise FileNotFoundError(f"ディレクトリ '{source_dir}' が見つからないか、ディレクトリではありません")

        # サポートするファイル拡張子を全て取得
        all_extensions = []
        for ext_list in rag_service.document_processor.SUPPORTED_EXTENSIONS.values():
            all_extensions.extend(ext_list)

        # ファイルを検索
        files = []
        for ext in all_extensions:
            files.extend(list(source_directory.glob(f"**/*{ext}")))

        logger.info(f"ディレクトリ '{source_dir}' 内に {len(files)} 個のファイルが見つかりました")

        # 差分処理の場合、ファイルレジストリを読み込む
        if incremental:
            file_registry = rag_service.document_processor.load_file_registry(processed_dir)
            logger.info(f"ファイルレジストリから {len(file_registry)} 個のファイル情報を読み込みました")

            # 処理対象のファイルを特定
            files_to_process = []
            for file_path in files:
                str_path = str(file_path)
                # ファイルのメタデータを取得
                current_metadata = rag_service.document_processor.get_file_metadata(str_path)

                # レジストリに存在しない、またはハッシュ値が変更されている場合のみ処理
                if (
                    str_path not in file_registry
                    or file_registry[str_path]["hash"] != current_metadata["hash"]
                    or file_registry[str_path]["mtime"] != current_metadata["mtime"]
                    or file_registry[str_path]["size"] != current_metadata["size"]
                ):
                    files_to_process.append(file_path)
                    # レジストリを更新
                    file_registry[str_path] = current_metadata

            print(f"処理対象のファイル数: {len(files_to_process)} / {len(files)}")

            # 各ファイルを処理
            for i, file_path in enumerate(files_to_process):
                try:
                    file_results = rag_service.document_processor.process_file(
                        str(file_path), processed_dir, chunk_size, overlap
                    )
                    results.extend(file_results)
                    processed_files += 1
                    print(
                        f"処理中... {processed_files}/{len(files_to_process)} ファイル ({(processed_files / len(files_to_process) * 100):.1f}%): {file_path}"
                    )
                except Exception as e:
                    logger.error(f"ファイル '{file_path}' の処理中にエラーが発生しました: {str(e)}")
                    # エラーが発生しても処理を続行
                    continue

            # ファイルレジストリを保存
            rag_service.document_processor.save_file_registry(processed_dir, file_registry)
        else:
            # 差分処理でない場合は全てのファイルを処理
            for i, file_path in enumerate(files):
                try:
                    file_results = rag_service.document_processor.process_file(
                        str(file_path), processed_dir, chunk_size, overlap
                    )
                    results.extend(file_results)
                    processed_files += 1
                    print(
                        f"処理中... {processed_files}/{total_files} ファイル ({(processed_files / total_files * 100):.1f}%): {file_path}"
                    )
                except Exception as e:
                    logger.error(f"ファイル '{file_path}' の処理中にエラーが発生しました: {str(e)}")
                    # エラーが発生しても処理を続行
                    continue

            # 全ファイル処理の場合も、新しいレジストリを作成して保存
            file_registry = {}
            for file_path in files:
                str_path = str(file_path)
                file_registry[str_path] = rag_service.document_processor.get_file_metadata(str_path)
            rag_service.document_processor.save_file_registry(processed_dir, file_registry)

        logger.info(f"ディレクトリ '{source_dir}' 内のファイルを処理しました（合計 {len(results)} チャンク）")
        return results

    # 進捗表示付きの処理メソッドに置き換え
    rag_service.document_processor.process_directory = process_directory_with_progress

    # インデックス化を実行
    result = rag_service.index_documents(directory_path, processed_dir, chunk_size, chunk_overlap, incremental)

    # 元のメソッドに戻す
    rag_service.document_processor.process_directory = original_process_directory

    if result["success"]:
        incremental_text = "差分" if incremental else "全て"
        logger.info(
            f"インデックス化が完了しました（{incremental_text}のファイルを処理、{result['document_count']} ドキュメント、{result['processing_time']:.2f} 秒）"
        )
        print(
            f"インデックス化が完了しました（{incremental_text}のファイルを処理）\n"
            f"- ドキュメント数: {result['document_count']}\n"
            f"- 処理時間: {result['processing_time']:.2f} 秒\n"
            f"- メッセージ: {result.get('message', '')}"
        )
    else:
        logger.error(f"インデックス化に失敗しました: {result.get('error', '不明なエラー')}")
        print(
            f"インデックス化に失敗しました\n"
            f"- エラー: {result.get('error', '不明なエラー')}\n"
            f"- 処理時間: {result['processing_time']:.2f} 秒"
        )
        sys.exit(1)


def get_document_count():
    """
    インデックス内のドキュメント数を取得する
    """
    logger = setup_logging()
    logger.info("インデックス内のドキュメント数を取得しています...")

    # 環境変数の読み込み
    load_dotenv()

    # RAGサービスの作成
    rag_service = create_rag_service_from_env()

    # ドキュメント数を取得
    try:
        count = rag_service.get_document_count()
        logger.info(f"インデックス内のドキュメント数: {count}")
        print(f"インデックス内のドキュメント数: {count}")
    except Exception as e:
        logger.error(f"ドキュメント数の取得中にエラーが発生しました: {str(e)}")
        print(f"ドキュメント数の取得中にエラーが発生しました: {str(e)}")
        sys.exit(1)


def main():
    """
    メイン関数

    コマンドライン引数を解析し、適切な処理を実行します。
    """
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(
        description="MCP RAG Server CLI - インデックスのクリアとインデックス化を行うためのコマンドラインインターフェース"
    )
    subparsers = parser.add_subparsers(dest="command", help="実行するコマンド")

    # clearコマンド
    subparsers.add_parser("clear", help="インデックスをクリアする")

    # indexコマンド
    index_parser = subparsers.add_parser("index", help="ドキュメントをインデックス化する")
    index_parser.add_argument(
        "--directory",
        "-d",
        default=os.environ.get("SOURCE_DIR", "./data/source"),
        help="インデックス化するドキュメントが含まれるディレクトリのパス",
    )
    index_parser.add_argument("--chunk-size", "-s", type=int, default=500, help="チャンクサイズ（文字数）")
    index_parser.add_argument("--chunk-overlap", "-o", type=int, default=100, help="チャンク間のオーバーラップ（文字数）")
    index_parser.add_argument("--incremental", "-i", action="store_true", help="差分のみをインデックス化する")

    # countコマンド
    subparsers.add_parser("count", help="インデックス内のドキュメント数を取得する")

    args = parser.parse_args()

    # コマンドに応じた処理を実行
    if args.command == "clear":
        clear_index()
    elif args.command == "index":
        index_documents(args.directory, args.chunk_size, args.chunk_overlap, args.incremental)
    elif args.command == "count":
        get_document_count()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()