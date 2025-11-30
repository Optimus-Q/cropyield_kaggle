#!/bin/bash

cat << "EOF"
   _____                   _____ _____    _____                             
  / ____|                 |_   _|  __ \  |  __ \                            
 | |     _ __ ___  _ __     | | | |  | | | |__) |   _ _ __  _ __   ___ _ __ 
 | |    | '__/ _ \| '_ \    | | | |  | | |  _  / | | | '_ \| '_ \ / _ \ '__|
 | |____| | | (_) | |_) |  _| |_| |__| | | | \ \ |_| | | | | | | |  __/ |   
  \_____|_|  \___/| .__/  |_____|_____/  |_|  \_\__,_|_| |_|_| |_|\___|_|   
                  | |                                                       
                  |_|                                                                                                     
EOF



# crop id runnner argument parameters
mode=$1
configyamlfile=$2

# train and test mode
if [ $mode == 'train' ]
then
   echo "--------------Training CROP ID---------------"
   python train.py $configyamlfile
elif [ $mode == 'test' ]
then
   echo "--------------Testing CROP ID---------------"
   python test.py $configyamlfile
fi 
