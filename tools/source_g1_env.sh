#!/usr/bin/env bash

set -a

if [ -f "${HOME}/g1_ws/.env" ]; then
    source "${HOME}/g1_ws/.env"
fi

set +a
