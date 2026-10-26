#!/bin/bash
# box.sh '<cmd>' [limit]: run a command on metatron over telnet, print only its output. Returns as soon as the command
# ends (its end marker seen), or after `limit` seconds (default 120). Before 2026-10-26 it always waited the full limit.
# nc's output goes straight to a file (a `tr` in between buffered it until exit); the \r is stripped at the end.
cmd="$1"; w=${2:-120}
tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
( { sleep 0.5; printf 'echo @@S""TART; %s; echo @@E""ND\r\n' "$cmd"; sleep "$w"; } | timeout $((w+5)) nc 192.168.88.6 23 > "$tmp" 2>/dev/null ) &
bg=$!; disown "$bg"
for ((i = 0; i < w * 10; i++)); do grep -q $'^@@END\r$' "$tmp" 2>/dev/null && break; sleep 0.1; done
# the pipeline's processes (the sleep, nc), by pid, deepest first
kids() { for p in $(pgrep -P "$1"); do kids "$p"; echo "$p"; done; }
kill $(kids "$bg") "$bg" 2>/dev/null
tr -d '\r' < "$tmp" | sed -n '/^@@START$/,/^@@END$/p' | sed '1d;$d'
