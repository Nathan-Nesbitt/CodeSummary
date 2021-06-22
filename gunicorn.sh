#!/bin/bash
gunicorn main:server --timeout=5000 -w 1 --threads 4 -b 0.0.0.0:3000 