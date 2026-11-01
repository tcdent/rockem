#!/bin/bash
# bmc-thermal-account.sh — create the thermal daemon's own account on metatron's BMC and the credential
# file the box reads. Run by tcdent on elsenor (needs sudo for the admin env). Idempotent: an existing
# 'thermal' account gets a fresh password.
#
#   bringup/box/thermal/bmc-thermal-account.sh
#
# Writes bundles/secrets/metatron-bmc-thermal.env (mode 644: the bundles export is ro with root_squash,
# so the box reads it as nobody), then checks the account can PATCH the fan endpoint by writing the
# current ratios back unchanged.
set -euo pipefail
cd "$(dirname "$0")/../../.."
eval "$(sudo cat /etc/ipmi/metatron-bmc-admin.env | grep -E '^IPMI_')"
B="https://$IPMI_HOST"; A=(-u "$IPMI_USER:$IPMI_PASSWORD")
ROLE=${ROLE:-Operator}
PW=$(python3 -I -c 'import secrets,string; a=string.ascii_letters+string.digits; print("Tm"+"".join(secrets.choice(a) for _ in range(14))+"9!")')

id=$(curl -sk -m 15 "${A[@]}" "$B/redfish/v1/AccountService/Accounts" | python3 -I -c '
import sys,json,subprocess
for m in json.load(sys.stdin)["Members"]: print(m["@odata.id"])' | while read -r u; do
  n=$(curl -sk -m 15 "${A[@]}" "$B$u" | python3 -I -c 'import sys,json; print(json.load(sys.stdin).get("UserName"))')
  [ "$n" = thermal ] && basename "$u"
done)
body="{\"UserName\":\"thermal\",\"Password\":\"$PW\",\"RoleId\":\"$ROLE\",\"Enabled\":true}"
if [ -n "$id" ]; then
  et=$(curl -sk -m 15 "${A[@]}" -D - -o /dev/null "$B/redfish/v1/AccountService/Accounts/$id" | awk 'tolower($1)=="etag:"{print $2}' | tr -d '\r')
  code=$(curl -sk -m 30 "${A[@]}" -X PATCH "$B/redfish/v1/AccountService/Accounts/$id" -H 'content-type: application/json' -H "If-Match: $et" -d "{\"Password\":\"$PW\",\"RoleId\":\"$ROLE\",\"Enabled\":true}" -o /tmp/bmc-acct.$$ -w '%{http_code}')
else
  code=$(curl -sk -m 30 "${A[@]}" -X POST "$B/redfish/v1/AccountService/Accounts" -H 'content-type: application/json' -d "$body" -o /tmp/bmc-acct.$$ -w '%{http_code}')
fi
echo "account: http $code"; python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1])); print((d.get("error") or {}).get("message","ok"))' /tmp/bmc-acct.$$ || true; rm -f /tmp/bmc-acct.$$
case $code in 200|201|204) ;; *) echo "account not created; nothing written"; exit 1;; esac

mkdir -p bundles/secrets
umask 022
cat > bundles/secrets/metatron-bmc-thermal.env <<EOF
# metatron BMC account for the thermal daemon (thermal/, docs/queue/101-fan-control.md), role $ROLE.
# Made by bringup/box/thermal/bmc-thermal-account.sh; read by the box over the bundles share.
IPMI_HOST=172.16.66.6
IPMI_USER=thermal
IPMI_PASSWORD=$PW
EOF
chmod 644 bundles/secrets/metatron-bmc-thermal.env
echo "wrote bundles/secrets/metatron-bmc-thermal.env"

# can the account drive the fans? write the current ratios back unchanged
P="$B/redfish/v1/Chassis/1/Oem/Public/Thermal"; T=(-u "thermal:$PW")
cur=$(curl -sk -m 15 "${T[@]}" "$P" -D /tmp/bmc-h.$$)
et=$(awk 'tolower($1)=="etag:"{print $2}' /tmp/bmc-h.$$ | tr -d '\r'); rm -f /tmp/bmc-h.$$
fans=$(echo "$cur" | python3 -I -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps([{"PwmSensorName":f["PwmSensorName"],"SpeedRatio":f["SpeedRatio"]} for f in d["Fans"]]))')
code=$(curl -sk -m 30 "${T[@]}" -X PATCH "$P" -H 'content-type: application/json' -H "If-Match: $et" -d "{\"ControlMode\":\"Manual\",\"Fans\":$fans}" -o /dev/null -w '%{http_code}')
echo "fan write as thermal ($ROLE): http $code"
[ "$code" = 200 ] || echo "the $ROLE role cannot write the fans: rerun with ROLE=Administrator"
