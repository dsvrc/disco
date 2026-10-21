#!/bin/bash
# Server management script for web_demo (uses current conda environment)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="$SCRIPT_DIR/server.pid"
LOG_FILE="$SCRIPT_DIR/app.log"

case "$1" in
    start)
        # Check if already running
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "Server is already running (PID: $PID)"
                echo "Use '$0 restart' to restart, or '$0 stop' to stop first."
                exit 0
            else
                echo "Removing stale PID file"
                rm -f "$PID_FILE"
            fi
        fi

        # Also check if port 5000 is occupied
        if fuser 5000/tcp > /dev/null 2>&1; then
            echo "Port 5000 is already in use. Killing existing process..."
            fuser -k 5000/tcp 2>/dev/null
            sleep 1
        fi

        echo "Starting web server..."
        echo "  Python: $(which python)"
        echo "  Log:    $LOG_FILE"

        nohup python app.py > "$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        sleep 3

        if ps -p "$(cat "$PID_FILE")" > /dev/null 2>&1; then
            echo "Server started (PID: $(cat "$PID_FILE"))"
            echo "Access:"
            echo "  http://localhost:5000      - Standard mode"
            echo "  http://localhost:5000/mixer - Live mixer mode"
        else
            echo "Failed to start server. Check log:"
            tail -20 "$LOG_FILE"
            rm -f "$PID_FILE"
            exit 1
        fi
        ;;

    stop)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "Stopping server (PID: $PID)..."
                kill "$PID"
                sleep 2
                # Force kill if still alive
                if ps -p "$PID" > /dev/null 2>&1; then
                    kill -9 "$PID"
                fi
                echo "Server stopped"
            else
                echo "Server is not running (stale PID file)"
            fi
            rm -f "$PID_FILE"
        else
            echo "No PID file found."
        fi

        # Clean up port 5000 as fallback
        if fuser 5000/tcp > /dev/null 2>&1; then
            echo "Killing remaining process on port 5000..."
            fuser -k 5000/tcp 2>/dev/null
        fi
        ;;

    restart)
        bash "$SCRIPT_DIR/server.sh" stop
        sleep 1
        bash "$SCRIPT_DIR/server.sh" start
        ;;

    status)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "Server is running (PID: $PID)"
                curl -s http://localhost:5000/api/status 2>/dev/null || echo "(API not responding)"
            else
                echo "Server is not running (stale PID file)"
                rm -f "$PID_FILE"
            fi
        else
            echo "Server is not running"
        fi
        ;;

    log)
        tail -50 "$LOG_FILE"
        ;;

    *)
        echo "Usage: $0 {start|stop|restart|status|log}"
        exit 1
        ;;
esac
