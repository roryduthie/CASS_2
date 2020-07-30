#!/bin/sh
source venv/bin/activate
exec gunicorn -b :8200 --access-logfile - --error-logfile - app --timeout 3000
