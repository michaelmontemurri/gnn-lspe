#!/bin/bash

DIR=data/molecules/
mkdir -p $DIR
cd $DIR

FILE=ZINC.pkl
if test -f "$FILE"; then
    echo -e "$FILE already downloaded."
else
    echo -e "\nDownloading $FILE..."
    curl -L -o ZINC.pkl https://data.dgl.ai/dataset/benchmarking-gnns/ZINC.pkl
fi
