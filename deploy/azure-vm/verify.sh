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

wait_for_url "application readiness" \
    curl --fail --silent --show-error http://127.0.0.1:8000/readyz -o /dev/null

anonymous_status="$(curl --silent --show-error --output /dev/null \
    --write-out '%{http_code}' "https://${public_host}/workbench")"
if [[ "$anonymous_status" != "401" ]]; then
    echo "Expected anonymous HTTPS status 401, received ${anonymous_status}." >&2
    exit 1
fi

auth_user="$(cut -d: -f1 "$secret_root/basic-auth.txt")"
auth_password="$(cut -d: -f2- "$secret_root/basic-auth.txt")"
wait_for_url "authenticated HTTPS ingress" \
    curl --fail --silent --show-error --user "${auth_user}:${auth_password}" \
        "https://${public_host}/workbench" -o /dev/null

data_volume_before="$(${compose[@]} config --volumes | grep '^rag-mvp-data$')"
test "$data_volume_before" = "rag-mvp-data"
${compose[@]} restart app
wait_for_url "application readiness after restart" \
    curl --fail --silent --show-error http://127.0.0.1:8000/readyz -o /dev/null
docker run --rm --mount type=volume,src=rag-mvp-data,dst=/data,readonly \
    alpine:3.22 test -s /data/metadata.sqlite3

echo "Verification passed for https://${public_host}"
