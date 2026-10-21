#!/bin/bash
echo "Cleaning up ports 5000 and 8765..."


fuser -k 5000/tcp 2>/dev/null
fuser -k 8765/tcp 2>/dev/null


PID_8765=$(netstat -tlnp | grep :8765 | awk '{print $7}' | cut -d'/' -f1)
PID_5000=$(netstat -tlnp | grep :5000 | awk '{print $7}' | cut -d'/' -f1)

[ ! -z "$PID_8765" ] && kill -9 $PID_8765
[ ! -z "$PID_5000" ] && kill -9 $PID_5000

sleep 2

echo "Starting servers..."

cd "$(dirname "$0")"
nohup python audio_capture_server.py > audio.log 2>&1 &
nohup python app.py > web.log 2>&1 &

echo "Check status with: netstat -tlnp | grep -E '5000|8765'"