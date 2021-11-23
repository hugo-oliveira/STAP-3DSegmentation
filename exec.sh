#!/bin/bash

# train full
python3 train.py "./configs/$1" > "$2"
  
config_cerebellum="$(expr match "$1" '\(.*\)\.yaml')cerebellum.yaml"
config_vent="$(expr match "$1" '\(.*\)\.yaml')stem_ventricle.yaml"

#split for spec 1 and 2
python3 crop_specialist.py "./configs/$1" >> "$2"

# spec 1
echo "============================spec_cerebellum=====================\n" >>   "$2"
python3 train.py "./configs/$config_cerebellum" >> "$2"

# spec 2
echo "============================spec_ventricle=====================\n" >>  "$2"
python3 train.py "./configs/$config_vent" >> "$2"

# Join results
echo "============================join=====================\n" >>  "$2"
python3 join.py "./configs/$1"  "./configs/$config_cerebellum" "./configs/$config_vent" >> "$2"
