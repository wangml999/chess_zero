#!/bin/bash

counter=1
while [ $counter -le 10 ]
do
	echo start self playing...
	for i in {1..20}; do ../bin/chess_r -s -n 100 -log; done

	echo start training...
	python train.py
	
	((counter++))
done


