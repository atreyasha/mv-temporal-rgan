#!/bin/bash

# move pre-commit hook into local .git folder for activation
cp ./hooks/pre-commit.sample ./.git/hooks/pre-commit

# get lfw faces data
cd ./data && ./lfw_setup.sh && cd ..
