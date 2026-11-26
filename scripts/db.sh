#!/usr/bin/env bash
# PostgreSQL + pgvector local au projet (données dans .pgdata, port 5433, écoute locale uniquement).
# Usage : scripts/db.sh start | stop | status
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PGBIN="${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}"
DATA="$ROOT/.pgdata"
PORT="${PGPORT_MAINTIDOC:-5433}"

case "${1:-start}" in
  start)
    if [ ! -f "$DATA/PG_VERSION" ]; then
      "$PGBIN/initdb" -D "$DATA" -U maintidoc --auth=trust -E UTF8 >/dev/null
    fi
    "$PGBIN/pg_ctl" -D "$DATA" status >/dev/null 2>&1 || \
      "$PGBIN/pg_ctl" -D "$DATA" -l "$DATA/server.log" -w \
        -o "-p $PORT -c listen_addresses=127.0.0.1 -k ''" start >/dev/null
    "$PGBIN/psql" -h 127.0.0.1 -p "$PORT" -U maintidoc -d postgres -tAc \
      "SELECT 1 FROM pg_database WHERE datname='maintidoc'" | grep -q 1 || \
      "$PGBIN/createdb" -h 127.0.0.1 -p "$PORT" -U maintidoc maintidoc
    "$PGBIN/psql" -h 127.0.0.1 -p "$PORT" -U maintidoc -d maintidoc -qc \
      "CREATE EXTENSION IF NOT EXISTS vector"
    echo "PostgreSQL prêt sur 127.0.0.1:$PORT (base maintidoc)"
    ;;
  stop)   "$PGBIN/pg_ctl" -D "$DATA" stop ;;
  status) "$PGBIN/pg_ctl" -D "$DATA" status ;;
  *) echo "Usage : $0 start|stop|status" >&2; exit 1 ;;
esac
