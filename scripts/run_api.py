import logging
import sys
from pathlib import Path

import uvicorn


# 启动 FastAPI 服务。
# 运行后可以访问：
# - http://127.0.0.1:8030/health
# - http://127.0.0.1:8030/docs
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def setup_logging() -> None:
    """给应用日志配输出通道。

    uvicorn 只配置自己的 uvicorn.* 日志器，应用里各模块的
    logging.getLogger(__name__) 会一路传播到 root——而 root 默认没有
    handler，INFO/WARNING 全部进黑洞。曾经排查 judge 全 0 事故时，
    服务端的 warning 一条都看不见，只能靠接口返回反推。

    这里给 root 挂基础通道：应用日志立即可见；uvicorn 自己的日志器
    propagate=False 且自带 handler，不会被重复输出。httpx/httpcore/
    openai 这类三方库的 INFO 刷屏压到 WARNING。
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        stream=sys.stdout,
    )
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main():
    setup_logging()
    uvicorn.run(
        "app.api:app",
        host="127.0.0.1",
        port=8030,
        reload=False,
    )


if __name__ == "__main__":
    main()
