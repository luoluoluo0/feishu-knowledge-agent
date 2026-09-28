# feishu-knowledge-agent

把管理员指定的飞书共享文件夹自动同步到 Milvus，并通过 LangGraph Agent
提供带原文链接的中文问答。支持飞书新版 Docx、PDF、WebSocket 事件、每小时
递归对账、SQLite 幂等任务队列、失败重试和 7 天软删除。

> 本项目是社区开源项目，与飞书或 Lark 官方无隶属、授权或背书关系。

## 数据流

```text
飞书共享文件夹
  → WebSocket 事件 + 定时递归对账
  → SQLite 去重任务与同步台账
  → 独立 Worker
  → Docx Blocks / PDF MinerU 或 pypdf
  → 父子切块 + Embedding + Milvus
  → Agent 检索 + 飞书原文链接
```

首版使用应用身份，配置的所有文件夹内容会进入同一个共享知识库；不实现
用户级 OAuth 和 ACL。不要把权限不同的私密资料放入同步文件夹。

Agent 默认按通用飞书知识库助手工作，可回答制度、流程、产品、项目和业务
文档问题。原项目的论文检索、文献卡片、PPT 与组会提纲能力作为兼容场景保留；
`search_paper` 等内部工具名暂不重命名，以免破坏旧调用和已有部署。

## 1. 创建飞书应用

1. 在飞书开放平台创建企业自建应用。
2. 为应用开通云空间文件元数据只读、云文档内容读取、文件下载以及事件订阅
   所需权限；发布新版本并等待管理员审批。
3. 在目标文件夹的协作者设置中添加该应用，让应用至少拥有查看权限。
4. 在“事件与回调”中选择长连接模式，订阅文件创建、编辑、标题更新、删除和
   移入回收站事件。
5. 从文件夹 URL 取得 folder token，填入 `FEISHU_FOLDER_TOKENS`。多个 token
   用英文逗号分隔。

权限名称可能随飞书控制台调整；以接口页“权限要求”显示的名称为准。程序若
权限不足，会在管理接口保留飞书 Log ID，并自动降级到定时对账。

## 2. 本地启动

官方支持 Python 3.11。Windows 建议使用独立的 3.11 虚拟环境，避免系统
Anaconda 中 `pyarrow` DLL 与 `pymilvus` 发生 ABI 冲突。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

填写 `.env` 中的模型、飞书和 API Key。随后启动 Milvus、API 与独立 Worker：

```powershell
docker compose -f docker-compose.milvus.yml up -d
python scripts/run_api.py
python -m app.feishu_sync.cli watch
```

也可以把三者一起放进 Docker：

```bash
docker compose up -d --build
```

需要自托管 Langfuse 时，再显式启用可选 profile，并先替换 `.env` 中的所有
Langfuse 数据库、加密和初始化密码：

```bash
docker compose --profile langfuse up -d --build
```

## 3. 首次验证

先只扫描，不改 Milvus：

```powershell
python -m app.feishu_sync.cli sync --once
```

再由 Worker 处理当前待执行任务：

```powershell
python -m app.feishu_sync.cli worker --once
```

查看运行状态（所有管理接口都需要业务 API Key）：

```powershell
curl.exe -H "X-API-Key: $env:SERVICE_API_KEY" `
  http://127.0.0.1:8030/admin/feishu-sync/status
```

最后打开 `http://127.0.0.1:8030`，询问刚同步文档里的独有关键词；来源卡片应
出现“打开飞书原文”。

## CLI

```text
python -m app.feishu_sync.cli sync --once    # 完整递归扫描并排队
python -m app.feishu_sync.cli worker --once  # 处理当前到期任务后退出
python -m app.feishu_sync.cli watch          # 启动扫描、Worker、事件和每小时对账
```

生产部署建议 API 和 `watch` 分成两个进程。`watch` 内的 Worker 使用 SQLite
30 分钟租约并每分钟续租；异常退出后任务会被重新领取。失败按 1 分钟、5 分钟、
30 分钟、2 小时退避，最多尝试 5 次。

## 管理 API

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/admin/feishu-sync/status` | 最近扫描、源健康、积压、失败与语料版本 |
| GET | `/admin/feishu-sync/documents` | 文档同步状态与原文链接 |
| GET | `/admin/feishu-sync/jobs` | 任务、重试次数与错误 |
| POST | `/admin/feishu-sync/run` | 后台触发完整对账 |
| POST | `/admin/feishu-sync/jobs/{job_id}/retry` | 重试 failed 任务 |
| POST | `/ask/stream` | SSE 流式问答 |

`SERVICE_API_KEY` 是首选变量；旧部署中的 `FEISHU_SERVICE_API_KEY` 仍兼容。

## 同步语义

- 同一文件内容 SHA-256 未变化时不会重复生成向量。
- 对账时会按 item_id 抽查 Milvus 实际行数是否与本地镜像一致；向量库被重建
  或误删导致台账与数据脱节时，自动清空内容哈希并复活任务走重灌。
- 文档更新采用“保存旧快照 → 生成新块/向量 → 替换 → 查回质检”；失败会删除
  新块并恢复旧块。
- 事件明确删除时立即排队移出 Milvus；扫描首次未发现为
  `suspect_missing`，连续两次才转 `soft_deleted`。
- 软删除保留 7 天，期间重新出现会自动恢复；过期后清理本地快照。
- 每次成功新增、更新或停用都会递增 `corpus_revision`，查询缓存和父块存储
  无需重启即可看到新内容。

## 测试

```bash
python -m pytest -q --basetemp=.pytest-tmp
python -m ruff check app tests scripts
```

真实飞书凭证测试只允许本地显式运行，不进入公共 CI。Milvus 集成测试由 CI 的
独立 Docker job 执行。合成样例位于 `examples/`。

## 已知限制

- 只支持 Docx 和 PDF；图片、附件只保留占位，扫描 PDF 需要安装 MinerU。
- SQLite 任务队列面向单实例 Worker，不适合横向扩容；后续可替换为 Redis/Celery。
- 不实现用户级飞书权限映射；PPTX、Sheets、Bitable 和 Wiki 留待后续版本。
- MinerU 不存在时 `FEISHU_PDF_PARSER=auto` 会回退到 pypdf，扫描件可能无法提取。

## License

[MIT](LICENSE)
