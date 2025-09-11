#!/bin/bash

COMMAND='
sleep 10
cd /home/DS_ws 
python3 main.py
'

gnome-terminal -- bash -c "$COMMAND; exec bash"