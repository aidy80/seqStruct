#!/bin/bash

cd OG

scp -r afike01@login.cluster.tufts.edu:/cluster/tufts/ylin12/aidan/cyclic/6mers/Neighbor_analysis/thermoFiles/enth/* .

for file in *; do
	if [ "${#file}" -eq 10 ]; then
		mv $file ${file:4:6}
	fi
	echo $file
done
cd ..
