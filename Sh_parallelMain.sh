#!/bin/bash

totNumParams=4
count=0
threadCap=2
pwd

for ((i = 0;i <= $totNumParams;i++))
do
	if [ $i -eq 0 ]; then
		python main.py $i 
		echo "Outputting Params"
	else 
		count=$((count+1))
		if [ $count -lt $threadCap ] 
		then
			python main.py $i $count &
		else
			python main.py $i $count
			count=0
		fi
		echo $count
	fi
done
