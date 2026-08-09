#!/usr/bin/env bash
set -euo pipefail

public_host="${1:?public host is required}"
app_root="/opt/rag-mvp/app"
secret_root="/opt/rag-mvp/secrets"
compose=(docker compose --env-file "$app_root/deployment.env" \
    -f "$app_root/compose.yaml" -f "$app_root/deploy/azure-vm/compose.azure.yaml")

wait_for_url() {
    local description="$1"
    shift
    local attempt
    for attempt in $(seq 1 24); do
        if "$@"; then
            return 0
        fi
        sleep 5
    done
    echo "$description did not become ready within 120 seconds." >&2
    return 1
}

anonymous_https_protected() {
    local status
    status="$(curl --silent --show-error --output /dev/null \
        --write-out '%{http_code}' "https://${public_host}/workbench")" || return 1
    [[ "$status" == "401" ]]
}

wait_for_url "application health" \
    curl --fail --silent --show-error http://127.0.0.1:8000/healthz -o /dev/null

readiness_file="$(mktemp)"
trap 'rm -f "$readiness_file"' EXIT
readiness_status="$(curl --silent --show-error --output "$readiness_file" \
    --write-out '%{http_code}' http://127.0.0.1:8000/readyz)"
if [[ "$readiness_status" != "200" ]] && ! grep -q '"reason":"index_not_ready"' "$readiness_file"; then
    echo "Application readiness failed for a reason other than an empty index." >&2
    exit 1
fi

wait_for_url "anonymous HTTPS protection" anonymous_https_protected

auth_user="$(cut -d: -f1 "$secret_root/basic-auth.txt")"
auth_password="$(cut -d: -f2- "$secret_root/basic-auth.txt")"
wait_for_url "authenticated HTTPS ingress" \
    curl --fail --silent --show-error --user "${auth_user}:${auth_password}" \
        "https://${public_host}/workbench" -o /dev/null

"${compose[@]}" config --volumes | grep -qx 'app_data'
docker volume inspect rag-mvp-data >/dev/null
"${compose[@]}" restart app
wait_for_url "application health after restart" \
    curl --fail --silent --show-error http://127.0.0.1:8000/healthz -o /dev/null
docker run --rm --mount type=volume,src=rag-mvp-data,dst=/data,readonly \
    alpine:3.22 test -s /data/metadata.sqlite3

echo "Verification passed for https://${public_host}"
