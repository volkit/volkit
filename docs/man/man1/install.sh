#!/bin/bash
WD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
install -g 0 -o 0 -m 0644 $WD/vkt.1 /usr/local/man/man1/
install -g 0 -o 0 -m 0644 $WD/vkt-render.1 /usr/local/man/man1/
gzip /usr/local/man/man1/vkt.1
