#!/bin/sh
# backend/entrypoint.sh

# Exit immediately if a command exits with a non-zero status.
set -e

# Correctly parse the host and port from the DATABASE_URL
host=$(echo $DATABASE_URL | awk -F'[@/:]' '{print $4}')
port=$(echo $DATABASE_URL | awk -F'[@/:]' '{print $5}')

echo "Waiting for postgres at $host:$port..."

# Use a loop to wait until we can successfully connect to the database
until PGPASSWORD=$POSTGRES_PASSWORD psql -h "$host" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c '\q'; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 1
done

>&2 echo "Postgres is up - starting server"

# Execute the main uvicorn command
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
