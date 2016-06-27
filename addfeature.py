# -*- coding: utf-8 -*-
from random import randint
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import numpy as np
#from matplotlib import style
from sklearn.cluster import KMeans
job_num=15#assign this around 15 
n_nodes =200
n_targets =60
r_sensing =40	
alpha = 0.25 
E=1
delta_E = 0.1*E
beta=0.35	
W=5

zeroes_column=[0 for x in xrange(n_nodes)]
computation_para=[0 for x in xrange(n_nodes)]
target_mapping=[]
Tu_initial=0
Tu=[0 for x in xrange(n_nodes)]

TargetSU=[0 for x in xrange(n_nodes)]
#TargetSU=[]
targetval=n_targets
target_mapping=[0 for x in xrange(n_targets)]
x_sensor=[0 for x in xrange(n_targets)]
y_sensor=[0 for x in xrange(n_targets)]

targets=[[0 for y in xrange(2)]for x in xrange(n_targets)]
nodes=[[0 for y in xrange(2)]for x in xrange(n_nodes)]

connections=[[0 for y in xrange(n_targets)]for x in xrange(n_nodes)]


for i in range (0,n_nodes):
	nodes[i][0]=randint(0,500)
	nodes[i][1]=randint(0,500)
	
for i in range (0,n_targets):
	targets[i][0]=randint(0,500)
	targets[i][1]=randint(0,500)
	
for i in range (0,n_targets):
	error=True
	for j in range (0,n_nodes):
		if r_sensing**2 > ( (targets[i][0]-nodes[j][0])**2 + (targets[i][1] - nodes[j][1] )**2 ) :
			connections[j][i]=1
			error=False
	if (error):
		print "Target " + str(i) + " is not being covered"

 
x_target = [x[0] for x in targets]
y_target = [x[1] for x in targets]
x_node = [x[0] for x in nodes]
y_node = [x[1] for x in nodes]


Eu=[E for x in xrange(n_nodes)]
node_sensor=[False for x in xrange (n_nodes)]
#Assuming that all the targets are being covered
while targetval>0 :	#Waiting till all the targets have been covered
	for i in range(0,n_nodes):
		for j in range(0,n_targets):
			TargetSU[i]=int(TargetSU[i])+int(connections[i][j])	
		#Assuming the sensor energy does not change
		Tu[i]=( (1-alpha*Eu[i]/E ) - (beta*TargetSU[i]/n_targets) )*W - Tu_initial
	Tu_initial=min(Tu)
	#print (Tu)
	while (min(Tu)==Tu_initial):	#This is for multiple occurences of the same waiting time
		sensor_index=Tu.index(Tu_initial)	#Index of the sensor whose waiting time got over
		for k in range (0,n_targets):
			if( connections[sensor_index][k] ==1):
				Eu[sensor_index]=Eu[sensor_index]-delta_E
				for p in range (0,n_nodes):
					connections[p][k]=zeroes_column[p]#Removing that target from connections
				targetval=targetval-1
			 	target_mapping[k]=sensor_index #Alloting targets their sensors
			 	node_sensor[sensor_index]=True
				x_sensor[k]=nodes[sensor_index][0]
				y_sensor[k]=nodes[sensor_index][1]
		Tu[sensor_index]=W
jobset_arr=[ 0 for x in xrange(job_num)]#total num of jobs , here we are considering just the jobs , not the fact that each job has two parts sensing and processing etc , but we can do that as well
jobset_device=[0 for x in xrange(job_num)]
device_CPU=[0 for x in xrange(n_nodes)]
for i in range(n_nodes):
	if(node_sensor[i]==true):
		device_CPU[i]=randint(25,75)#we gave every so called mobile phone a CPU availability 	
for i in range(job_num):	
	jobset_arr[i]=randint(20,70)#not giving other parameters like tn right now as we are not even considering the jobs' priority , can e done if asked later

#note one more thing we are gonna assign any first node that matches the computation power of the job, this is done for no other reason than to make it easy to code and as there are no other parameters required right now ... 
#if other parameters like tn for devices and priority for jobs , then the device assignment will be different and will follow some kind of formula/algo instead of this first come first serve relation  
for i in range(job_num):
	for j in range(n_nodes):
		if(node_sesnor[j]):
			if(device_CPU[j]>=jobset_arr[i]):
				device_CPU[j]=device_CPU[j]-jobset_arr[i]
				jobset_device[i]=j
				break
#we have assigned every job to a device now to bring in the concept of machine learning 


# Convert to a set and back into a list.
set = set(jobset_device)
result = list(set)
cost_arr=[0 for x in xrange(len(result)-1)]

device_cordinate_arr=[[0 for y in xrange(2)]for x in xrange(len(result))]
for i in range(len(result)):
	device_cordinate_arr[i][0]=nodes[result[i]][0]
	device_cordinate_arr[i][1]=nodes[result[i]][1]

for n_clusters in range(1,len(result)):
	kmeans=KMeans(n_clusters)
	kmeans.fit(device_cordinate_arr)
	centroids=kmeans.cluster_centers_
	labels = kmeans.labels_
	req_val=0
	node_relay=[]
	node_relay_label=[]
#K means algo used not according to the cost function mentioned in the paper. Maybe yes.
	for i in range(n_clusters-1):
		distance_arr1=[0 for x in xrange(n_nodes)]
		distance_arr2=[0 for x in xrange(n_nodes)]
		for j in range(n_nodes):
			if(node_sensor[j]):
				distance_arr2[j]=( (centroids[i,0]- node[j][0])**2 + (centroids[i,1]- nodes[j][1])**2  )**0.5
			else:
				distance_arr1[j]=( (centroids[i,0]- node[j][0])**2 + (centroids[i,1]- nodes[j][1])**2  )**0.5 #distance formula for centroids with i and nodes with j
			
		req_val=np.argmax(distance_arr1)

		for k in range(n_nodes):
			if(distance_arr1[k]==0):
				distance_arr1[k]=distance_arr1[req_val]
		node_relay.append(np.argmin(distance_arr1))
		req_val=np.argmax(distance_arr2)
		for k in range(n_nodes):
			if(distance_arr2[k]==0):
				distance_arr2[k]=distance_arr2[req_val]
		node_relay_label,append(np.argmin(distance_arr2)
		
	
	
	for n in range(n_clusters)
		for m in range(len(result))			
			if(labels[m]==node_relay_label[n])
				cost_arr[n_clusters]=cost_arr[n_clusters]+( (node[m][0]- node[m][0])**2 + (nodes[][1]- nodes[m][1])**2  )**0.5
				
clusternum=np.argmin(cost_arr)
#we now have the required cluster num perform k means again to get the final result and display
        

