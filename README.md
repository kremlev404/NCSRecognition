### NCS Recognition Project
### Performed by Anton Kremlev
### Verified setups:
* Ubuntu 20:
1) Openvino 2021.4.582
* Raspbian buster:
1) Raspberry 3b+
2) Openvino 2020_1 //models 2019 R3
### Build:
1) mkdir build && cd build
2) cmake -B . -S ..
3) execute makeDataTree.sh to create data folder
4) go to data folder and follow guide's in each folder 
### Neural Calculation Performed by NCS2
### Models can be downloaded here: https://download.01.org/opencv/ or via downloader
### Args Example: -args_include=false -d_type=4