# 失物宇宙 · 月球失物招领所

一把正在下雨的伞，三个奇怪的候选主人。你是新来的管理员：查看线索、作出判断，让物品回家。越少线索通关，星数越高。全套剧情与数据为本项目原创。

## 验证状态

已在本机 Dify 1.19.0 成功导入、发布并调用 DeepSeek。seekdb 1.4.0 承担关卡档案和玩家进度存储。完整实测记录见 evidence/dify-live.json，续玩截图见 evidence/dify-resumed.jpg。部署补充步骤见 DEPLOY.md。

## 技术分工

`Dify Workflow 开始 → JSON编码 → HTTP调用 → 状态检查 → LLM叙事 → 结果与数据库回执`

- Dify 承担输入、HTTP 调用、异常检查、模型调用和结果输出，入口直接使用 Dify 发布的应用页。
- seekdb 通过 MySQL 协议保存 `lost_cases` 物品档案和 `lost_players` 玩家进度。每次行动真实读档、锁定玩家状态并更新；重启后端或重新打开 Dify 页面仍能继续。不是 Dify 部署数据库，也不宣称向量检索。
- 后端决定线索顺序、通关和星数；未公开答案不传入模型。结果同时展示原始回执，模型文案不是权威判定。
- 不使用 PowerContext、Embedding、Rerank 或自定义插件。

## 环境与配置

本机环境：Python 3.11、Dify 1.19.0（官方 tag，commit 8e689e868b63b6dd2b449d0eb568ca5cc7127759）、seekdb Server 1.4.0。Python 依赖版本见 requirements.txt。Dify 安装官方市场的 langgenius/deepseek 0.0.26 插件，并在模型供应商页面配置 DeepSeek API Key；DSL 使用 deepseek-flash。模型调用按 DeepSeek 账户额度计费。

seekdb 请按官方文档 https://docs.oceanbase.com/obd/deploy-seekdb 部署，创建或选择 `test` 数据库。不要拿普通 MySQL 的成功运行当作指定技术验收。保留 `SELECT VERSION()` 结果及实际 seekdb 安装版本。

本机已安装 Docker Desktop 4.99.0，Docker 引擎 29.9.3。可使用已部署的远程 seekdb；若在另一台 Linux 机器部署，请让后端能访问数据库 2881 端口。建议为数据库预留至少 2 核、4GB 内存，并以所选版本官方资源要求为准。

在本目录使用 PowerShell：

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
$env:SEEKDB_HOST='实际数据库地址'
$env:SEEKDB_PORT='2881'
$env:SEEKDB_USER='root'
$env:SEEKDB_PASSWORD='实际数据库密码'
$env:SEEKDB_DATABASE='test'
$env:GAME_TOKEN='自行生成的长随机令牌'
.venv/Scripts/python init_db.py
.venv/Scripts/python -m unittest -v
.venv/Scripts/python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

`.env.example` 只作配置参考，应用不会自动加载 `.env`。Linux 使用 `export KEY=value` 设置相同变量，虚拟环境命令改为 `.venv/bin/python`。初始化可重复执行，不会清空玩家状态。

## 导入 Dify（仅一个文件）

1. 工作室 → 导入 DSL → 选择 `workflow.yml`。
2. 在 HTTP 节点把 URL 改为 Dify worker 能访问的后端 `/turn` 地址；在工作流环境变量中把 GAME_TOKEN 的占位值替换为后端 GAME_TOKEN（secret 类型）。不要提交修改后含真实令牌的 DSL。
3. 本地 Docker Desktop Dify 使用 `host.docker.internal:8000`；Linux 容器需要主机网关或同一网络服务名。Dify Cloud 无法访问你电脑的 localhost，需要部署 HTTPS 后端。本地 Dify 的 docker/.env 设置 `SSRF_PROXY_ALLOW_PRIVATE_DOMAINS=host.docker.internal` 后运行 `docker compose up -d`，仅允许该后端域名。
4. 重新绑定 LLM 模型和供应商。先运行调试，再发布应用。每次使用相同玩家代号执行一个行动。
5. 先看数据库回执，再看故事。HTTP 非 200 会中止，不生成虚构成功信息。

## 完整验收样例

初始条件：初始化成功，使用随机且独有的玩家代号如 `judge_20261109_a1`。执行下列行动时始终使用同一代号。

|顺序|行动|预期回执|
|---|---|---|
|1|开始|下雨的伞、三个候选人；clues=0，done=false|
|2|线索|制造雨的标签；clues=1|
|3|归还 白羽|认领失败；clues 仍为1|
|4|重新打开应用，再选线索|出现薄荷与温室；clues=2，证明读取之前进度|
|5|归还 小满|成功、2颗星、月球薄荷茶；done=true|
|6|线索|提示本轮结束|
|7|开始|恢复 clues=0、done=false，可重复测试|

运行前后用数据库客户端查询：

```sql
SELECT VERSION();
SELECT id, payload FROM lost_cases WHERE id='moon';
SELECT id, state FROM lost_players WHERE id='judge_20261109_a1';
```

保存 Dify HTTP 节点输入输出、最终故事与回执截图、以上 SQL 结果及时间到 `evidence/`。不得只保存连接成功截图。已交付的实际结果见 evidence/dify-live.json（包含初次沙箱配置失败及修复后成功记录）。

## 已知限制

仅一个原创关卡；玩家代号是私密进度口令，不是登录账号，知道代号的人可重置该进度，只适合比赛隔离试玩。请使用随机代号，勿用于真实身份或敏感数据。公开上线前需绑定可信用户身份并加配额。

数据库写入先于 LLM 调用；若模型失败，行动可能已经保存。不要自动重试 HTTP 节点。请先查玩家状态或输入「开始」重开。本稿没有事务跨越 LLM 的保证。

## 提交到比赛仓库

测评平台导入官方仓库后应出现 `2026-competition-ecosystem` 分支。把本目录复制到该仓库的 `dify/lost-universe/`，不要复制本地 `.venv`、令牌或缓存。在正式仓库内确认基线：

```sh
git fetch origin
git switch 2026-competition-ecosystem
git status
# 复制资产、完成真实验收后再暂存与提交
git add dify/lost-universe
git commit -m "Add Lost Universe Dify game with seekdb state"
git rev-parse HEAD
```

若 origin 是私有导入仓库，应确认该分支确实继承官方赛事分支历史。不要在无历史的空仓库新建同名分支冒充赛事基线。把最终 commit hash 按平台要求填写。

## 参考

- 赛事基线：https://github.com/oceanbase/powercontext/tree/2026-competition-ecosystem
- Dify API与后端密钥原则：https://docs.dify.ai/en/api-reference/guides/get-started
- seekdb：https://github.com/oceanbase/seekdb

代码与剧情未复制参考游戏；上述项目用于接口与部署参考。



