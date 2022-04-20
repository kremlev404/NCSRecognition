#!/bin/bash

mkdir -p "$(pwd)/data/db"
mkdir -p "$(pwd)/data/video"
mkdir -p "$(pwd)/data/models"
cd data
touch README.md
echo "### Create in db tree like: username/(user_photos with .png extension)" > README.md
echo "### Put in video your videos.mp4 if you want to analyze them" >> README.md
echo "### Put in models your models like: model/FP16/model.(xml:bin)" >> README.md
echo "File Tree Created"

