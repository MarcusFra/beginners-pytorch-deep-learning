#!/usr/bin/env bash
cd /app
waitress-serve --call 'catfish_server_trial:app'