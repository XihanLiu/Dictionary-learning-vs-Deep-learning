#!/usr/bin/python
import os
import random
dataset_path = r'/Users/liuxihan/Desktop/compressed sensing and sparse recovery/CroppedYale'
dataset_list_path = r'/Users/liuxihan/Desktop/compressed sensing and sparse recovery/Dictionary-learning-vs-Deep-learning'
with open(os.path.join(dataset_list_path,'train.txt'),'w') as train:
	with open(os.path.join(dataset_list_path,'test.txt'),'w') as test:
		for dataset_dir in  os.listdir(dataset_path):
			num_test = 0
			for sample in os.listdir(os.path.join(dataset_path,dataset_dir)):
				if '.pgm' in sample and '.bad' not in sample:
					flag = random.randint(0,1)
					if flag == 1 & num_test < 20:
						test.write(sample+' '+sample[5:7]+'\n')
						num_test+=1
	
					else:
						train.write(sample+' '+sample[5:7]+'\n')
train.close()
test.close()	
