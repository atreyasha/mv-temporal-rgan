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
    read -rep "specify dimensionality of lfw-faces (28 is recommended): " ans2
    python3 pre-process-faces.py --dim "$ans2" --out "lfw_$ans2.npy"
fi
