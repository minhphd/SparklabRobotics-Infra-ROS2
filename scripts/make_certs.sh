#!/usr/bin/env bash
# Generate the self-signed TLS cert the Quest bridge serves to the headset.
#
#     ./scripts/make_certs.sh                 # cover every local address
#     ./scripts/make_certs.sh 10.195.64.139   # ...plus/only the ones you name
#
# Every local IPv4 goes in the subjectAltName, so one cert works on the lab LAN,
# over the campus link and over an `adb reverse` USB tunnel. The SAN is the
# whole point — a cert without one is invalid for every name and the Quest
# browser rejects it outright. Regenerate when the machine gains an address.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Physical interfaces only: docker/veth/bridge addresses just pad the cert.
mapfile -t AUTO < <(ip -4 -o addr show scope global \
    | grep -vE ' (docker|veth|br-|virbr)' \
    | awk '{split($4,a,"/"); print a[1]}')
IPS=("${@:-}"); [ -z "${IPS[0]:-}" ] && IPS=("${AUTO[@]}")
[ ${#IPS[@]} -gt 0 ] || { echo "no addresses found; pass one explicitly" >&2; exit 1; }

SAN="DNS:localhost,IP:127.0.0.1"
for ip in "${IPS[@]}"; do SAN="$SAN,IP:$ip"; done

if [ -f cert.pem ]; then
    ts=$(date +%Y%m%d%H%M%S)
    cp cert.pem "cert.pem.bak-$ts"; cp key.pem "key.pem.bak-$ts"
    echo "[certs] backed up existing pair -> *.bak-$ts"
fi

openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
    -keyout key.pem -out cert.pem \
    -subj "/CN=${IPS[0]}" -addext "subjectAltName=$SAN" 2>/dev/null
chmod 600 key.pem

echo "[certs] wrote cert.pem / key.pem"
openssl x509 -in cert.pem -noout -subject -ext subjectAltName | sed 's/^/  /'
echo
echo "The headset must accept this cert once (advanced -> proceed). Regenerating"
echo "invalidates that acceptance, so do not re-run it casually."
