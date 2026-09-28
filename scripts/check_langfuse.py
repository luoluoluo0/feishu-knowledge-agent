import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from app.observability import check_langfuse_connection, shutdown_observability


def main() -> int:
    try:
        if check_langfuse_connection():
            print("Langfuse 凭据与网络连接正常。")
            return 0
        print("Langfuse 连接检查未通过。")
        return 1
    except Exception as exc:
        print(f"Langfuse 连接检查失败：{exc}")
        return 1
    finally:
        shutdown_observability()


if __name__ == "__main__":
    raise SystemExit(main())
