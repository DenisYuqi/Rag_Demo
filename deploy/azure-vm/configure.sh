#!/usr/bin/env bash
set -euo pipefail

public_host="${1:?public host is required}"
source_revision="${2:?source revision is required}"
app_root="/opt/rag-mvp/app"
secret_root="/opt/rag-mvp/secrets"
auth_user="ragadmin"
caddy_image="caddy:2.10.2-alpine@sha256:4c6e91c6ed0e2fa03efd5b44747b625fec79bc9cd06ac5235a779726618e530d"

test -s "$secret_root/provider-key"
chown 10001:10001 "$secret_root/provider-key"
chmod 0400 "$secret_root/provider-key"

if [[ ! -s "$secret_root/basic-auth.txt" ]]; then
    auth_password="$(openssl rand -base64 32 | tr -d '\r\n=/+')"
    printf '%s:%s\n' "$auth_user" "$auth_password" > "$secret_root/basic-auth.txt"
    chmod 0400 "$secret_root/basic-auth.txt"
fi

auth_password="$(cut -d: -f2- "$secret_root/basic-auth.txt")"
auth_hash="$(docker run --rm "$caddy_image" caddy hash-password --plaintext "$auth_password")"

PUBLIC_HOST="$public_host" BASIC_AUTH_USER="$auth_user" BASIC_AUTH_HASH="$auth_hash" \
APP_ROOT="$app_root" SOURCE_REVISION="$source_revision" python3 - <<'PY'
import os
from pathlib import Path

app_root = Path(os.environ["APP_ROOT"])
template = (app_root / "deploy/azure-vm/Caddyfile").read_text(encoding="utf-8")
runtime = (
    template.replace("{{PUBLIC_HOST}}", os.environ["PUBLIC_HOST"])
    .replace("{{BASIC_AUTH_USER}}", os.environ["BASIC_AUTH_USER"])
    .replace("{{BASIC_AUTH_HASH}}", os.environ["BASIC_AUTH_HASH"])
)
(app_root / "deploy/azure-vm/Caddyfile.runtime").write_text(runtime, encoding="utf-8")

revision = os.environ["SOURCE_REVISION"]
deployment_env = f"""RAG_MVP_IMAGE=rag-mvp:{revision}
RAG_MVP_APP_VERSION=0.1.0
RAG_MVP_SOURCE_REVISION={revision}
RAG_MVP_BIND_ADDRESS=127.0.0.1
RAG_MVP_HOST_PORT=8000
RAG_MVP_CONTAINER_PORT=8000
RAG_MVP_DATA_VOLUME=rag-mvp-data
RAG_MVP_MODEL_CACHE_VOLUME=rag-mvp-model-cache
RAG_MVP_OPENAI_API_KEY_SECRET_FILE=/opt/rag-mvp/secrets/provider-key
RAG_MVP_PROVIDER_BACKEND=openai
RAG_MVP_BGE_PROFILE_ENABLED=false
RAG_MVP_DEFAULT_RETRIEVAL_PROFILE=openai-api
RAG_MVP_WORKBENCH_ENABLED=true
RAG_MVP_LOG_LEVEL=INFO
"""
(app_root / "deployment.env").write_text(deployment_env, encoding="utf-8")
PY

chmod 0600 "$app_root/deployment.env" "$app_root/deploy/azure-vm/Caddyfile.runtime"
echo "Runtime configuration created for https://${public_host}"
echo "Basic Auth credential retained at ${secret_root}/basic-auth.txt"
