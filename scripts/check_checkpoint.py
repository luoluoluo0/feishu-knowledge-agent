from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from app.config import get_settings


def main():
    settings = get_settings()
    checkpoint_path = Path(settings.checkpoint_db_path)

    print("Checkpoint 数据库路径：", checkpoint_path)
    print("Checkpoint 数据库是否存在：", checkpoint_path.exists())

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: F401
    except ImportError:
        print("当前状态：未安装 langgraph-checkpoint-sqlite，会回退到 InMemorySaver")
        print("安装命令：python -m pip install langgraph-checkpoint-sqlite")
        return

    print("当前状态：已安装 langgraph-checkpoint-sqlite，可以使用 SQLite Checkpointer")


if __name__ == "__main__":
    main()
