#!/bin/bash

mkdir -p models

fileid="1yUbF9Z0CdMPGyR068RpN97KiWp0bjGLX"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o models/yolov7-tiny_256x320.onnx

fileid="1tw7fOd8blxoYIj6t2gEjehb6yKFsv7om"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o models/yolov7-tiny_640x640.onnx


fileid="1uMwi-EiRLE2o_i-DTQ_pDYFJvFGv_ant"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o models/yolov7_256x320.onnx


fileid="1fHWqoiVDk5l7rNTppA7Gl1vHUaxh_2Mo"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o models/yolov7_640x640.onnx



echo Download finished.