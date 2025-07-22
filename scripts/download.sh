#!/bin/bash

links=( "https://github.com/boreng0817/IFCap/releases/download/v1.0/evaluation.zip"
    "https://github.com/boreng0817/IFCap/releases/download/v1.0/annotations.zip" )

for link in ${links[@]};
do
    wget $link;
done

for f in annotations.zip evaluation.zip;
do
    unzip $f
done

rm *.zip
