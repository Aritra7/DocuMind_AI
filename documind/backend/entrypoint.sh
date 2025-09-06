#!/bin/sh
# backend/entrypoint.sh

set -e  # exit if anything fails

echo "Running entrypoint script…"

# Parse DB host/port out of DATABASE_URL
host=$(echo $DATABASE_URL | awk -F'[@/:]' '{print $4}')
port=$(echo $DATABASE_URL | awk -F'[@/:]' '{print $5}')

echo "Waiting for postgres at $host:$port..."

until PGPASSWORD=$POSTGRES_PASSWORD psql -h "$host" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c '\q'; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 1
done

>&2 echo "Postgres is up - starting server"

# Start FastAPI through uvicorn
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
