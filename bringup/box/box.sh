#!/bin/bash
# box.sh '<cmd>' [wait]: run a command on metatron over telnet, print only its output
cmd="$1"; w=${2:-2}
(sleep 0.5; printf 'echo @@S""TART; %s; echo @@E""ND\r\n' "$cmd"; sleep $w) | timeout $((w+5)) nc 192.168.88.6 23 2>/dev/null | tr -d '\r' | sed -n '/^@@START$/,/^@@END$/p' | sed '1d;$d'
