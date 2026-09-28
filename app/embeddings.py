import logging
import re
import time

from langchain_openai import OpenAIEmbeddings

from app.config import Settings


logger = logging.getLogger(__name__)

CJK_PATTERN = re.compile(r"[一-鿿]")

# 标点在这套词表里大多没有对应项，会被拆成多个 token。
PUNCTUATION_PATTERN = re.compile(r"[.,;:()\[\]{}\"'`~!?@#$%^&*+=<>/\\|_–—‘’“”…]")

# 每类字符平均消耗多少 token，由实测阈值反推：
#   纯中文 500 字通过、520 字拒绝   -> 512 / 500 ≈ 1.02
#   纯英文 1300 字符通过、1500 拒绝 -> 512 / 1300 ≈ 0.39
#   标点取 1.3：实测有标点密度 52% 的 LaTeX 片段（\frac{1}{\frac{1}{…）
#   在系数 1.0 下仍被拒，说明这类符号的 token 消耗被低估了。
#
# 这些系数是从现有语料标定的经验值，不是精确测量。
# 真正的保险是 milvus_store 里的逐条降级——撞上边界只会丢个别块，
# 不会让整批写入失败。
TOKEN_PER_CJK_CHAR = 1.02
TOKEN_PER_LATIN_CHAR = 0.39
TOKEN_PER_PUNCT_CHAR = 1.3

# bge-large-zh-v1.5 的序列上限。
MAX_TOKENS = 512
# 留一点余量，避免卡在边界上。
TOKEN_SAFETY_RATIO = 0.94

# 两个端点值，由上面的 token 模型导出，供测试和文档引用。
# 纯中文约 471 字，纯英文约 1234 字（都不含标点）。
# 中间地带按每类字符的 token 消耗加权，见 estimate_char_limit()。
EMBED_MAX_CHARS_ZH = int(MAX_TOKENS * TOKEN_SAFETY_RATIO / TOKEN_PER_CJK_CHAR)
EMBED_MAX_CHARS_EN = int(MAX_TOKENS * TOKEN_SAFETY_RATIO / TOKEN_PER_LATIN_CHAR)


def estimate_char_limit(text: str) -> int:
    """估算这段文本的安全字符上限。

    做法是先算「平均每个字符消耗多少 token」，再用 512 除以它。

    **不能用线性插值。** 字符上限和中文占比是倒数关系，不是线性关系：

        中文占比   线性插值   正确值
           0%       1300      1313
          50%        890       726   ← 差 164 个字
         100%        480       502

    中英混排的文本（中文论文里夹英文人名、机构名）正落在 50% 附近，
    用线性插值会严重高估，实测失败率接近 10%。所以按每类字符的
    token 消耗加权平均来算。

    标点单独给一个系数：它们在这套词表里大多没有对应项，
    会被拆成多个 token，实测密度 11% 的参考文献在高限值下会被拒。

    切分和建库共用这个函数，保证子块不会被 embedding 二次截断。
    """

    if not text:
        # 空文本没有 token 上限问题，返回最宽的值即可。
        return EMBED_MAX_CHARS_EN + 200

    total = len(text)
    cjk_chars = len(CJK_PATTERN.findall(text))
    punct_chars = len(PUNCTUATION_PATTERN.findall(text))
    latin_chars = max(0, total - cjk_chars - punct_chars)

    tokens_per_char = (
        cjk_chars * TOKEN_PER_CJK_CHAR
        + latin_chars * TOKEN_PER_LATIN_CHAR
        + punct_chars * TOKEN_PER_PUNCT_CHAR
    ) / total

    return max(120, int(MAX_TOKENS * TOKEN_SAFETY_RATIO / tokens_per_char))

# SiliconFlow 的 embedding 接口会偶发返回 400：
#   {'code': 20015, 'message': 'The parameter is invalid.'}
# 但同一段文本立刻重试通常就能成功，实测失败率约 8%。
# 这是上游抖动，不是参数真的有问题。
#
# OpenAI SDK 默认只重试 429 和 5xx，不会重试 400，所以在这里自己兜一层。
EMBEDDING_ATTEMPTS = 3
EMBEDDING_RETRY_DELAY_SECONDS = 0.5


class RetryingEmbeddings:
    """给查询向量化加一层重试。

    只包 embed_query —— 那是每次检索都要走的热路径，也是最容易撞上
    上游抖动的地方。其余属性通过 __getattr__ 透传，调用方可以把它当作
    普通的 embeddings 对象使用。
    """

    def __init__(
        self,
        inner: OpenAIEmbeddings,
        attempts: int = EMBEDDING_ATTEMPTS,
        delay: float = EMBEDDING_RETRY_DELAY_SECONDS,
    ):
        self._inner = inner
        self._attempts = attempts
        self._delay = delay

    def embed_query(self, text: str, **kwargs) -> list[float]:
        return self._retry(lambda: self._inner.embed_query(text, **kwargs))

    def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
        """批量向量化，用于建库。

        批量比逐条快一个数量级，但代价是抖动概率随批量放大：
        单条失败率约 8%，64 条一批几乎必挂，所以这里同样要重试。
        """

        return self._retry(lambda: self._inner.embed_documents(texts, **kwargs))

    def _retry(self, call):
        last_error: Exception | None = None

        for attempt in range(1, self._attempts + 1):
            try:
                return call()
            except Exception as exc:
                last_error = exc
                if attempt < self._attempts:
                    logger.warning(
                        "embedding 调用失败（第 %s/%s 次），%.1fs 后重试：%s",
                        attempt,
                        self._attempts,
                        self._delay,
                        exc,
                    )
                    time.sleep(self._delay)

        raise last_error

    def __getattr__(self, name: str):
        # _inner 本身要通过正常属性查找拿到，避免初始化期间递归。
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


def build_embeddings(settings: Settings) -> RetryingEmbeddings:
    """创建用于问题向量化的 embedding 模型。"""

    if not settings.silicon_api_key:
        raise ValueError("没有读取到 SILICON_API_KEY，请检查 .env")

    inner = OpenAIEmbeddings(
        model=settings.embeddings_model,
        api_key=settings.silicon_api_key,
        base_url=settings.embeddings_base_url,
        check_embedding_ctx_length=False,
    )
    return RetryingEmbeddings(inner)
