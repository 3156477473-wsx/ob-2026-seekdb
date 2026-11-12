# 本地部署补充（Windows + Docker Desktop）

已验证的组合：Dify 1.19.0、seekdb 1.4.0、Python 3.11、DeepSeek 插件 0.0.26。模型端到端验证状态以 README 和 evidence 为准。

## 1. 启动 seekdb

在 PowerShell 中运行，数据库端口仅绑定本机。以下无密码 root 仅用于本机隔离试玩；不要将端口公开到网络。

```powershell
docker volume create lost-universe-seekdb-data
docker run -d --name lost-universe-seekdb -p 127.0.0.1:2881:2881 -v lost-universe-seekdb-data:/var/lib/seekdb oceanbase/seekdb@sha256:4db4961869d554313520d6081fca29265ba6da6db7313311bd8acbed771a956b
```

等待数据库启动。已经存在容器时使用 `docker start lost-universe-seekdb`，不要重复创建或删除数据卷。

## 2. 启动 Dify

```powershell
git clone --branch 1.19.0 --depth 1 https://github.com/langgenius/dify.git dify-runtime
cd dify-runtime/docker
Copy-Item .env.example .env
```

编辑 `.env`：按官方模板生成独立随机密码/密钥，确保关联服务使用相同值。配置 `COMPOSE_PROFILES=weaviate,postgresql,collaboration`，`EXPOSE_NGINX_PORT=127.0.0.1:8080`，`EXPOSE_NGINX_SSL_PORT=127.0.0.1:8443`，`SSRF_PROXY_ALLOW_PRIVATE_DOMAINS=host.docker.internal`。`EXPOSE_PLUGIN_DEBUGGING_PORT` 保留纯数字 `5003`。设置 `APP_WEB_URL=http://localhost:8080`，让生成的体验链接使用正确端口。`SANDBOX_API_KEY` 与 `CODE_EXECUTION_API_KEY` 必须填写同一个随机值，否则代码节点返回 401。

为避免调试端口对外开放，新建 `docker-compose.override.yaml`：

```yaml
services:
  plugin_daemon:
    ports: !override
      - 127.0.0.1:5003:5003
```

运行 `docker compose up -d`，在 http://localhost:8080 初始化管理员。集成 → 模型供应商 → 安装官方深度求索插件 → 配置自己的 DeepSeek API Key。这里的 PostgreSQL/Weaviate 是 Dify 自身依赖，作品的物品和玩家进度实际保存在独立 seekdb 中。

## 3. 游戏服务及导入

回到本项目目录，按 README 的 Python 安装与环境变量步骤运行。当前本机试玩使用 `SEEKDB_HOST=127.0.0.1`、`SEEKDB_PORT=2881`、`SEEKDB_DATABASE=test`、`SEEKDB_USER=root`、`SEEKDB_PASSWORD` 为空字符串。

为 GAME_TOKEN 生成独立随机值，例如运行 `python -c "import secrets; print(secrets.token_urlsafe(32))"`，只在后端环境和 Dify secret 环境变量内保存。

Windows Docker Desktop 已验证可通过 `host.docker.internal:8000` 访问绑定于 `127.0.0.1:8000` 的本机后端。启动命令：

```powershell
.venv/Scripts/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

导入唯一 DSL `workflow.yml`，配置 GAME_TOKEN，检查 LLM 节点选择 DeepSeek 的可用模型，调试通过后发布。重启电脑后，启动 Docker Desktop、已有 seekdb 容器、Dify Compose 及 Python 后端；无需重新初始化或清空数据。

