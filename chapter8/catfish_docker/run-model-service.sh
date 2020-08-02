#!/usr/bin/env bash
cd /app
waitress-serve --call 'catfish_server:create_app'