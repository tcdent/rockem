import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", 6666))
with open("/home/slomp/Work/rockem/bringup/box/logs/metatron-console.log", "ab", buffering=0) as f:
    while True:
        d, _ = s.recvfrom(65536)
        f.write(d)
