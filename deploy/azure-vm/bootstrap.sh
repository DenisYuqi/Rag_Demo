#!/usr/bin/env bash
set -euo pipefail

admin_user="${1:?admin user is required}"

retry_twice() {
    local description="$1"
    shift
    local attempt
    for attempt in 1 2; do
        if "$@"; then
            return 0
        fi
        if [[ "$attempt" == "2" ]]; then
            echo "$description failed twice; stopping." >&2
            return 1
        fi
        echo "$description failed once; retrying." >&2
        sleep 5
    done
}

export DEBIAN_FRONTEND=noninteractive
retry_twice "apt metadata refresh" apt-get update
retry_twice "bootstrap package installation" \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg openssl

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

. /etc/os-release
architecture="$(dpkg --print-architecture)"
printf '%s\n' \
    "deb [arch=${architecture} signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list

retry_twice "Docker apt metadata refresh" apt-get update
retry_twice "Docker installation" \
    apt-get install -y --no-install-recommends \
        docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

systemctl enable --now docker
usermod -aG docker "$admin_user"
install -d -m 0755 -o "$admin_user" -g "$admin_user" /opt/rag-mvp/app
install -d -m 0700 -o root -g root /opt/rag-mvp/secrets

docker version --format '{{.Server.Version}}'
docker compose version
