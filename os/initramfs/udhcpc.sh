#!/bin/sh
# busybox udhcpc hook: apply the lease (address, router, resolvers) to $interface; keep its NTP servers for init.
case $1 in
  deconfig) ip addr flush dev "$interface" ;;
  bound|renew)
    ip addr flush dev "$interface"
    ip addr add "$ip/${mask:-24}" dev "$interface"
    [ -n "$router" ] && { ip route | grep -q '^default' || ip route add default via "${router%% *}" dev "$interface"; }
    [ -n "$dns" ] && { : > /etc/resolv.conf; for s in $dns; do echo "nameserver $s" >> /etc/resolv.conf; done; }
    [ -n "$hostname" ] && hostname "$hostname"
    [ -n "$ntpsrv" ] && for s in $ntpsrv; do echo "$s"; done >> /etc/ntp-servers ;;
esac
exit 0
