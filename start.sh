#!/bin/bash
# Start supervisord to launch Redis, Elasticsearch, and FastAPI
supervisord -c /app/supervisord.conf &

# Wait for Elasticsearch to boot up
echo "Waiting for Elasticsearch to start..."
until curl -s http://localhost:9200 >/dev/null; do
    sleep 2
done

# Run the database-to-elasticsearch sync script
echo "Running initial synchronization..."
cd /app/backend && python reindex.py

# Prevent the container from stopping
wait