#!/bin/sh

if [ -z "$1" ]
  then
    echo "Usage: $0 file.raw"
    exit
fi

volume=/tmp/volume.bin

vkt read --input $1 > $volume
vkt render < $volume

rm $volume
