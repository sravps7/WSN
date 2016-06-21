# -*- coding: utf-8 -*-
from random import randint
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import numpy as np
#from matplotlib import style
from sklearn.cluster import KMeans

n_nodes =350
n_targets =75
r_sensing =60	
alpha = 0.25 
E=1
delta_E = 0.1*E
beta=0.35	
W=5

zeroes_column=[0 for x in xrange(n_nodes)]

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



#number of clusters cannot exceed the number of sensors
#n_clusters=n_nodes-len(set(target_mapping))
#n_clusters=len(set(target_mapping))
totalcost=[0 for x in xrange(len(set(target_mapping))-1)]

for n_clusters in range(1,len(set(target_mapping))):
	kmeans=KMeans(n_clusters)
	kmeans.fit(nodes)
	centroids=kmeans.cluster_centers_
	labels = kmeans.labels_

#K means algo used not according to the cost function mentioned in the paper. Maybe yes.

	
#plt.scatter(centroids[:,0],centroids[:,1],marker ="x",s=150, linewidths=1, zorder=1)	
	node_relay=[False for x in xrange(n_nodes)]
	distance = [-1 for x in xrange(n_nodes)]

	sum_x=[0 for x in xrange (n_clusters)]
	sum_y=[0 for x in xrange (n_clusters)]
	number=[0 for x in xrange(n_clusters)]

	for i in range(n_nodes):
		if (node_sensor[i]):
			sum_x[labels[i]] = sum_x[labels[i]] + nodes[i][0]
			sum_y[labels[i]] = sum_y[labels[i]] + nodes[i][1]
			number[labels[i]] = number[labels[i]] + 1

	for i in range(n_clusters):
		if (number[i] > 0):
			sum_x[i]=sum_x[i]/number[i]
			sum_y[i]=sum_y[i]/number[i]


#plt.plot([row[0] for row in sensor_centroid], [row[1] for row in sensor_centroid] , 'r^')

	for j in range(len(target_mapping)) :	#For a sensor mapping target j
		for i in range(0,n_nodes):	# For a node i
			if(  ( labels[target_mapping[j] ] == labels[i] ) and (target_mapping[j]!= i)) : #Checking if node i is in the same cluster as a sensor mapping target j # also making sure sensor is not the node
				node_relay[i] = True
				distance[i] = ( (x_node[i]-sum_x[labels[i]])**2 + ( y_node[i] - sum_y[ labels[i]])**2 )**0.5
# Two arrays having distance from the nodes and whether the node is a potential sensor or not
	node_relay1=[False for x in xrange(n_nodes)]
	x_relay=[]
	y_relay=[]

#remaining nodes
	temp_var=[0 for x in xrange(n_clusters)]
	min_dist=[0 for x in xrange(n_clusters)]
	for i in range (0,n_clusters):	#iterating through all the clusters
		for k in range(0,n_nodes):
			if ( (labels[k]==i) and node_relay[k]):
				temp_var[i]=k	#gives the first potential relay node whose label is i 
				min_dist[i]=distance[ temp_var[i] ]	
				break		
			else:
				min_dist[i]=-1
	 	#min_dist initialised as the first instance of distance of the label i
		for j in range(n_nodes):	#iterating through all nodes  
			if(node_relay[j] and labels[j]==i):	#If node j is a potential relay and belongs to the cluster i
				if ( (distance[j] < min_dist[i]) and (min_dist[i] > 0) ):
					min_dist[i]=distance[j]		
					temp_var[i]=j

	for i in range (n_clusters):				
		if((min_dist[i]>=0)):
			node_relay1[temp_var[i]]= True
			x_relay.append(nodes[temp_var[i]][0])
			y_relay.append(nodes[temp_var[i]][1])
	for m in range(len(x_relay)):
		for k in range(n_nodes):
			if(nodes[k][0]==x_relay[m]): #we have found the relay nodes node number , now we can use labels to compute the distance and add them 
				tempvar1=k
		for j in range(len(x_sensor)):#add the code for x_sensor here 
			for k in range(n_nodes):
				if(nodes[k][0]==x_sensor[j]):
					tempvar2=k
			if(labels[tempvar1]==labels[tempvar2]):
				totalcost[n_clusters-1]=totalcost[n_clusters-1] + (nodes[tempvar1][0]-nodes[tempvar2][0])**2 + (nodes[tempvar1][1]-nodes[tempvar2][1])**2  #add the appropirate distance formula here , only distance squared used as the cost 

req_cluster_num=np.argmin(totalcost)+1#add proper formula here 
kmeans=KMeans(req_cluster_num)
kmeans.fit(nodes)
centroids=kmeans.cluster_centers_
labels = kmeans.labels_

#K means algo used not according to the cost function mentioned in the paper. Maybe yes.

	
#plt.scatter(centroids[:,0],centroids[:,1],marker ="x",s=150, linewidths=1, zorder=1)	
node_relay=[False for x in xrange(n_nodes)]
distance = [-1 for x in xrange(n_nodes)]

sum_x=[0 for x in xrange (req_cluster_num)]
sum_y=[0 for x in xrange (req_cluster_num)]
number=[0 for x in xrange(req_cluster_num)]

for i in range(n_nodes):
	if (node_sensor[i]):
		sum_x[labels[i]] = sum_x[labels[i]] + nodes[i][0]
		sum_y[labels[i]] = sum_y[labels[i]] + nodes[i][1]
		number[labels[i]] = number[labels[i]] + 1

for i in range(req_cluster_num):
	if (number[i] > 0):
		sum_x[i]=sum_x[i]/number[i]
		sum_y[i]=sum_y[i]/number[i]


#plt.plot([row[0] for row in sensor_centroid], [row[1] for row in sensor_centroid] , 'r^')

for j in range(len(target_mapping)) :	#For a sensor mapping target j
	for i in range(0,n_nodes):	# For a node i
		if(  ( labels[target_mapping[j] ] == labels[i] ) and (target_mapping[j]!= i)) : #Checking if node i is in the same cluster as a sensor mapping target j # also making sure sensor is not the node
			node_relay[i] = True
			distance[i] = ( (x_node[i]-sum_x[labels[i]])**2 + ( y_node[i] - sum_y[ labels[i]])**2 )**0.5
# Two arrays having distance from the nodes and whether the node is a potential sensor or not
node_relay1=[False for x in xrange(n_nodes)]
x_relay=[]
y_relay=[]

#remaining nodes
temp_var=[0 for x in xrange(req_cluster_num)]
min_dist=[0 for x in xrange(req_cluster_num)]
for i in range (0,req_cluster_num):	#iterating through all the clusters
	for k in range(0,n_nodes):
		if ( (labels[k]==i) and node_relay[k]):
			temp_var[i]=k	#gives the first potential relay node whose label is i 
			min_dist[i]=distance[ temp_var[i] ]	
			break		
		else:
			min_dist[i]=-1
	 	#min_dist initialised as the first instance of distance of the label i
	for j in range(n_nodes):	#iterating through all nodes  
		if(node_relay[j] and labels[j]==i):	#If node j is a potential relay and belongs to the cluster i
			if ( (distance[j] < min_dist[i]) and (min_dist[i] > 0) ):
				min_dist[i]=distance[j]		
				temp_var[i]=j

for i in range (req_cluster_num):				
	if((min_dist[i]>=0)):
		node_relay1[temp_var[i]]= True
		x_relay.append(nodes[temp_var[i]][0])
		y_relay.append(nodes[temp_var[i]][1])
	#else:
	#	node_relay[temp_var]=False		

#sravan read from here 
#after this point in the code we have the sensors plotted now we need to consider the crowd cloud of these sensors and store them so that if a job with some similar computation requirements arrive we can send it to this crowd cloud 
#we can consider each group (each group considered like a cloud ) and they can be identified from labels
#creating a new array and giving the devices random processing power and others
Pc=100#define appropirately
R=100 #define appropirately 
job_num=1
xnk=[[0 for y in xrange(job_num)]for x in xrange(n_nodes)]
ynk=[[0 for y in xrange(job_num)]for x in xrange(n_nodes)]

computation_para=[ 0 for x in xrange(req_cluster_num+1)]
device_dn=[[0 for y in xrange(5)]for x in xrange(n_nodes)]#{P p n , τ n , U n }and the last 2 for my own parameters, the last one are for xnk and  (which indicate the processing and sensing availability  check the paper
for i in range(n_nodes):
	if (node_sensor[i]):
		device_dn[i][0]=randint(0,100)#first parameter assignment ask about this  
		device_dn[i][1]=randint(0,100)#assignment of time 
		device_dn[i][2]=randint(0,50)#this assigns a random processing power to the sensor nodes
		
		device_dn[i][4]=0

jobset_arr=[[0 for y in xrange(6)]for x in xrange(job_num)]#P s k , t ks , u k , r k , s k , Priority k(check out from the paper ) 
#after this we apply give each job to a cloud based on the nearest best match (given in paper )
# we have only considered only one job because each of our cloud contain atmost three or four devices , previous code needs to be changed to make clouds bigger , in our case 
#we are going to give each active cloud a parameter at random which tells us about its computation power etc so as to choose from 
for i in range(job_num):
	jobset_arr[i][0]=randint(0,100)
	jobset_arr[i][1]=randint(1,5)
	jobset_arr[i][2]=randint(25,50)
	jobset_arr[i][3]=randint(1,5)
	jobset_arr[i][4]=randint(10,20)
	jobset_arr[i][5]=randint(0,1)
for i in range (0,req_cluster_num):
	for j in range(n_nodes):	
		if (labels[j]==i and node_sensor[j]):
			device_dn[j][3]=randint(50,100)
			computation_para[i]=computation_para[i] + device_dn[j][3]
#assignment of the so called weight parameter for computation power for choosing the cloud 
#in the next line we find the the cloud with the maximum computation parameter, the cloud_label parameter gives us the label number of the crowd cloud wchic is going to perform our job 

cloud_label=np.argmax(computation_para)
#taking out the devices that will be of our use and their parameters 
for k in range(job_num):
	if(jobset_arr[k][5]==1):#high priority 
#code wrt to what the algorithm
		time_array=[]
		for j in range(n_nodes):	
			if (labels[j]==cloud_label and node_sensor[j]):
				time_array.append(device_dn[j][1])
		for m in range(len(time_array)):
			temp_var1=0		
			for j in range(n_nodes):
				if (labels[j]==cloud_label and node_sensor[j]):
					if(device_dn[j][1]==time_array[m]):
						temp_var1=j
			print jobset_arr[k][2]
			if((device_dn[temp_var1][4]==0) and (100-device_dn[temp_var1][2]>=jobset_arr[k][2])):
				device_dn[temp_var1][4]=1
				#need to create an output variable for xnk 
				xnk[temp_var1][k]=1
				device_dn[temp_var1][2]=device_dn[temp_var1][2]+jobset_arr[k][2]
				print(device_dn[temp_var1][2])
				ynk[temp_var1][k]=1
				
				#need to assign an output array for ynk
#assign both sensing and processing jobs to one device like given in the algo 
	else:#low priority job
		
		energy_arr_small=[]
		energy_arr=[[0 for y in xrange(n_nodes)]for x in xrange(n_nodes)]#this is a 2d array required for the entering the variable m cross n for energy
		for i in range(n_nodes):#sensing device
			for j in range(n_nodes):#processing device
				if((node_sensor[i] and node_sensor[j])and (labels[i]==cloud_label and labels[j]==cloud_label)):#we have found the required device nodes , if they are able to get past this and reach here , then i and j are cloud nodes 
					if(i==j):#whether to exclude the Ec term or not , if the sensing and processing is done on the same device then there is no need to add that term since it is zero

						energy_arr[i][j]= (jobset_arr[k][0]*jobset_arr[k][1]) + (device_dn[j][0]*jobset_arr[k][2]*device_dn[j][1]*jobset_arr[k][3])      #some energy formula for i and j , here i is the sensing part and j is processing		
					else:	
						energy_arr[i][j]=(jobset_arr[k][0]*jobset_arr[k][1]) + (device_dn[j][0]*jobset_arr[k][2]*device_dn[j][1]*jobset_arr[k][3]) + Pc*jobset_arr[k][4]/R
					energy_arr_small.append(energy_arr[i][j])
							
		sorted(energy_arr_small, key=int)
		for m in range(len(energy_arr_small)):
			temp_var1=0
			temp_var2=0
			for n in range(n_nodes):
				for q in range(n_nodes):	
					if(energy_arr[n][q]==energy_arr_small[m]):
						temp_var1=n
						temp_var2=q
			if((device_dn[temp_var1][4]==0)and(100-device_dn[temp_var2][2]>=jobset_arr[k][2])):
				device_dn[temp_var1][4]=1#need to create an output variable for xnk 
				xnk[temp_var1][k]=1
				ynk[temp_var2][k]=1
				device_dn[temp_var2][2]=device_dn[temp_var2][2]+jobset_arr[k][2]
#need tocreate an output for ynk

plt.figure(1)
ax = plt.subplot(111)
plt.plot(x_node, y_node,'bs',markersize=10,label='nodes')
plt.plot(x_relay,y_relay,'g^',markersize=10,label='relay')
plt.plot(x_sensor,y_sensor,'rs',markersize=10,label='sensor')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.figure(2)
ax1 = plt1.subplot(111)
#plt1.plot(x_node, y_node,'bs',markersize=10,label='nodes')
plt1.plot(x_relay,y_relay,'g^',markersize=10,label='relay')
plt1.plot(x_sensor,y_sensor,'rs',markersize=10,label='sensor')
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
sum_xnk=0
sum_ynk=0

for i in range (n_nodes):
	for j in range(job_num):	
		sum_xnk=sum_xnk+ xnk[i][j]
		sum_ynk=sum_ynk + ynk[i][j]

print sum_xnk
print sum_ynk



#plt.legend(loc='upper left')
#plt.plot(sum_x, sum_y, 'g^')
#print "Blue squares are nodes"
#print "Red squares are sensors"
#print "Red triangles are targets"
#print "Green triangles are relays"
#print "Number of targets are " + str(n_targets)
#print "Number of sensors are " + str(len(set(target_mapping)))
#print "Number of clusters are " + str(n_clusters)  
#print "Number of relays are " + str(len(x_relay))
#if(req_cluster_num<len(set(target_mapping))):
#	print "yo"#check up if the code works lol
#plt.figure(1)
#plt.title("Number of targets are " + str(n_targets)+"  Number of sensors are  " + str(len(set(target_mapping)))+"  Number of relays are  " + str(len(x_relay)),fontsize=20)
#plt.figure(2)
#plt1.title("Number of targets are " + str(n_targets)+"  Number of sensors are  " + str(len(set(target_mapping)))+"  Number of relays are  " + str(len(x_relay)),fontsize=20)
#plt.show()

	
#Sufficient to make n_sensors number of clusters. This basically gives a set of sensors and relays with minimum distance from each other
#Some clusters might arise which do not contain any nodes. is it safe to discard these clusters
#cost = constant + distance to that node from the base station
#minimise cost to get the required relay nodes	

