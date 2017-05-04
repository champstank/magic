#!/bin/bash

docker pull derekdata/magic

docker run --rm -it -v $PWD:/data derekdata/magic:latest /bin/sh -c "python run.py $1 $2"
