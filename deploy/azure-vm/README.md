# Azure Global VM deployment

本目录包含将 RAG MVP 部署到 Azure Global 单台 Linux VM 的脚本和平台配置。完整的架构、
前置条件、安全设计、部署、更新、备份、费用与删除说明见
[Azure Global VM 部署技术文档](../../docs/deploy/azure-global-vm-deployment.zh-CN.md)。

本文档是可公开提交的操作模板，不记录任何真实订阅 ID、租户 ID、IP、域名、资源 ID、
SSH 私钥、Basic Auth 密码或 Provider API Key。示例中的 `<...>` 必须替换为目标环境的值，
且不应将替换后的私有交接信息提交到 Git。

## 部署文件

| 文件 | 用途 |
| --- | --- |
| `deploy.ps1` | Azure 预检、资源创建、源码上传、镜像构建和验证 |
| `bootstrap.sh` | 在 Ubuntu VM 上安装 Docker Engine、Buildx 和 Compose |
| `configure.sh` | 生成运行环境、Basic Auth 凭据和 Caddy 配置 |
| `compose.azure.yaml` | Azure 环境覆盖配置和 Caddy 服务 |
| `Caddyfile` | HTTPS、Basic Auth 和反向代理模板 |
| `verify.sh` | 健康、认证、容器重启和数据卷持久性验证 |

## 最小部署示例

先执行 `az login`，确认目标订阅，并在仓库外或被 Git 忽略的 `.env` 中配置
`RAG_MVP_OPENAI_API_KEY`。然后从项目根目录运行：

```powershell
& .\deploy\azure-vm\deploy.ps1 `
  -ResourceGroup '<RESOURCE_GROUP>' `
  -Location 'southeastasia' `
  -VmName 'vm-rag-mvp' `
  -VmSize 'Standard_D2as_v4' `
  -AdminUser '<ADMIN_USER>' `
  -DnsLabel '<GLOBALLY_UNIQUE_DNS_LABEL>' `
  -ProviderEnvFile '.env'
```

部署脚本会把 SSH 私钥保存到仓库外：

```text
%LOCALAPPDATA%\rag-mvp\azure-vm\<RESOURCE_GROUP>\id_ed25519
```

Provider key 通过 SSH 标准输入传输，不进入 Azure CLI 参数、镜像层或仓库。Basic Auth
凭据只保存在 VM 的受限文件中：

```text
/opt/rag-mvp/secrets/basic-auth.txt
```

不要将该文件内容、私钥或替换后的真实环境参数粘贴到 README、issue、日志或提交记录。

## 常用操作

连接 VM：

```powershell
$key = "$env:LOCALAPPDATA\rag-mvp\azure-vm\<RESOURCE_GROUP>\id_ed25519"
ssh -i $key '<ADMIN_USER>@<PUBLIC_HOST>'
```

在 VM 上管理 Compose 项目：

```bash
cd /opt/rag-mvp/app
compose=(docker compose --env-file deployment.env \
  -f compose.yaml -f deploy/azure-vm/compose.azure.yaml)

"${compose[@]}" ps
"${compose[@]}" up -d
"${compose[@]}" down
```

`docker compose down` 会保留 named volumes；不要在需要保留数据时添加 `--volumes`。
只停止容器不会停止 Azure VM 计费。暂时不用 VM 时从运维电脑执行：

```powershell
az vm deallocate --resource-group '<RESOURCE_GROUP>' --name 'vm-rag-mvp'
```

解除分配只停止 VM 计算费，磁盘和 Standard Public IP 仍可能计费。永久删除整个资源组会
删除 VM、磁盘、公网 IP、网络资源和 Docker 卷数据，必须先备份并再次确认目标：

```powershell
az group delete --name '<RESOURCE_GROUP>' --yes
```
