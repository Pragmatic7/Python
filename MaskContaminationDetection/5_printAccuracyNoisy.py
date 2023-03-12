from __future__ import division
import numpy as np
import csv

total = 0
trues = 0
accuracy = 0

with open('Results_MaskHeadandShoulders_noisy.txt', 'r') as f:
    for count, line in enumerate(f, start=1):
        if count % 2 == 0:
        	total = total + 1
        	print("total=", total)
        	# print line
        	tt = line.strip().split("\t")
        	print(float(tt[0]))
        	print(float(tt[1]))
        	if float(tt[0])<float(tt[1]):
        		trues = trues + 1
        		print("trues = ", trues)


accuracy = float(trues/total)
print('\nAccuracy of mask contamination detection in Noisy in percentage:', accuracy*100)

