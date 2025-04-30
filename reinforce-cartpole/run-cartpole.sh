#!/bin/sh
(echo "none"; echo "std") | parallel python main.py --lr=0.001 --gamma=0.99 --episodes=5000 --baseline={}
