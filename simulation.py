from random import randint
import matplotlib.pyplot as plt
import numpy as np
#from matplotlib import style
from sklearn.cluster import KMeans

n_nodes=500
n_targets=50
r_sensing=50	
alpha=0
Eu=1
E=1
beta=0.25
W=5
zeroes_column=[0 for x in xrange(n_nodes)]

target_mapping=[]
Tu_initial=0
Tu=[0for x in xrange(n_nodes)]
#Tu=[]
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
#plt.plot(x_target, y_target,'r^',x_node, y_node,'bs')



#Assuming that all the targets are being covered
while targetval>0 :	#Waiting till all the targets have been covered
	for i in range(0,n_nodes):
		for j in range(0,n_targets):
			TargetSU[i]=int(TargetSU[i])+int(connections[i][j])	
		#Assuming the sensor energy does not change
		Tu[i]=( (1-alpha*Eu/E ) - (beta*TargetSU[i]/n_targets) )*W - Tu_initial
	Tu_initial=min(Tu)
	#print (Tu)
	while (min(Tu)==Tu_initial):	#This is for multiple occurences of the same waiting time
		sensor_index=Tu.index(Tu_initial)	#Index of the sensor whose waiting time got over
		for k in range (0,n_targets):
			if( connections[sensor_index][k] ==1):
				for p in range (0,n_nodes):
					connections[p][k]=zeroes_column[p]#Removing that target from connections
				targetval=targetval-1
			 	target_mapping[k]=sensor_index #Alloting targets their sensors
				x_sensor[k]=nodes[sensor_index][0]
				y_sensor[k]=nodes[sensor_index][1]
		Tu[sensor_index]=W

#for i in range(0,n_targets):
#	circle=plt.Circle((x_sensor[i],y_sensor[i]),r_sensing,color='r')
#	fig = plt.gcf()
#	fig.gca().add_artist(circle)

#plt.plot(x_sensor,y_sensor,'rs',x_target,y_target,'r^')
plt.plot(x_sensor,y_sensor,'rs')
print "Red squares are sensors"
print "Red triangles are targets"

#number of clusters cannot exceed the number of sensors
#n_clusters=n_nodes-len(set(target_mapping))
n_clusters=len(set(target_mapping))
kmeans=KMeans(n_clusters)
kmeans.fit(nodes)
centroids=kmeans.cluster_centers_
labels = kmeans.labels_

#K means algo used not according to the cost function mentioned in the paper. Maybe yes.

for i in range (n_clusters):
	plt.plot(nodes[i][0],nodes[i][1],markersize=1)
	
plt.scatter(centroids[:,0],centroids[:,1],marker ="x",s=150, linewidths=1, zorder=1)	
node_relay=[False for x in xrange(n_nodes)]
distance = [-1 for x in xrange(n_nodes)]
sensor_centroid=[[0 for y in xrange(2)]for x in xrange(n_clusters)]
#for i in range(n_nodes-len(set(target_mapping))): #number of clusters, no of distinct labels

for m in range(0,n_clusters):
	counter=0
	temp_arr=[0 for x in xrange(len(target_mapping))]#define it to be zero everytime the loop runs
	
	for n in range(len(target_mapping)):
		if(labels[target_mapping[n]]==m):
 			temp_arr[n]=target_mapping[n]	
#figure something out for repitition of sensors 
	for e in range(len(temp_arr)):
		for f in range(e,len(temp_arr)):
			if(temp_arr[e]==temp_arr[f]):
				temp_arr[f]=0			
	for p in range(len(temp_arr)):
		if(temp_arr[p]!=0):
			counter=counter+1
	if(counter==0):
		sensor_centroid[m][0]=0
		sensor_centroid[m][1]=0
	else:
		for q in range(len(temp_arr)):
			if(temp_arr[q]!=0):
				sensor_centroid[m][0]=(sensor_centroid[m][0]+nodes[temp_arr[q]][0])/len(set(temp_arr))
				sensor_centroid[m][1]=(sensor_centroid[m][1]+nodes[temp_arr[q]][1])/len(set(temp_arr))



for j in range(len(target_mapping)) :	#For a sensor mapping target j
	for i in range(0,n_nodes):	# For a node i
		if(  ( labels[target_mapping[j] ] == labels[i] ) and (target_mapping[j]!= i)) : #Checking if node i is in the same cluster as a sensor mapping target j # also making sure sensor is not the node
			node_relay[i] = True
			distance[i] = ( (x_node[i]-sensor_centroid[labels[i]] [0])**2 + ( y_node[i] - sensor_centroid[ labels[i]][1]  )**2 )**0.5
# Two arrays having distance from the nodes and whether the node is a potential sensor or not

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
		node_relay[temp_var[i]]= True
		x_relay.append(nodes[i][0])
		y_relay.append(nodes[i][1])
	#else:
	#	node_relay[temp_var]=False		



#for i in range (0,n_nodes):
#	if(node_relay[i]==True):
#		x_relay.append(nodes[i][0])
#		y_relay.append(nodes[i][1])
	
plt.plot(x_relay,y_relay,'g^')
print "Green triangles are relays"
print "Number of targets are " + str(n_targets)
print "Number of sensors are " + str(len(set(target_mapping)))
print "Number of clusters are " + str(n_clusters)  
print "Number of relays are " + str(len(x_relay))
plt.show()
	
#Sufficient to make n_sensors number of clusters. This basically gives a set of sensors and relays with minimum distance from each other
#Some clusters might arise which do not contain any nodes. is it safe to discard these clusters
#cost = constant + distance to that node from the base station
#minimise cost to get the required relay nodes	

	
