#!/bin/bash
set -e

read -rep "create pre-commit hook for python dependencies? (y/n): " ans
if [ $ans == "y" ]; then
    # move pre-commit hook into local .git folder for activation
    cp ./hooks/pre-commit.sample ./.git/hooks/pre-commit
fi

read -rep "download and unzip lfw-faces data? (y/n): " ans
if [ $ans == "y" ]; then
    # get lfw-faces data
    cd ./data
    wget -O lfwcrop_grey.zip http://conradsanderson.id.au/lfwcrop/lfwcrop_grey.zip
    unzip -o lfwcrop_grey.zip
    cd ..
fi

# pre-process lfw-faces and save to numpy binary
read -rep "pre-process lfw-faces into numpy binary? (y/n): " ans
if [ $ans == "y" ]; then
    read -rep "specify kernel size for downsampling lfw-faces (2 is recommended): " ans2
    python3 pre-process-faces.py --kernel "$ans2"
fi
