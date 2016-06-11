from random import randint
import matplotlib.pyplot as plt
import numpy as np
#from matplotlib import style
from sklearn.cluster import KMeans

n_nodes =300
n_targets =50
r_sensing =50	
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

#for i in range(0,n_targets):
#	circle=plt.Circle((x_sensor[i],y_sensor[i]),r_sensing,color='r')
#	fig = plt.gcf()
#	fig.gca().add_artist(circle)

#plt.plot(x_sensor,y_sensor,'rs',x_target,y_target,'g^')


#number of clusters cannot exceed the number of sensors
#n_clusters=n_nodes-len(set(target_mapping))
n_clusters=len(set(target_mapping))
kmeans=KMeans(n_clusters)
kmeans.fit(nodes)
centroids=kmeans.cluster_centers_
labels = kmeans.labels_

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
		

p_alpha=1
p_beta=0
p_gamma=0
p_delta=0
ram=[0 for x in xrange(n_nodes)]
memory=[0 for x in xrange(n_nodes)]
distance=[0 for x in xrange(n_nodes)]
power_node=[-1 for x in xrange(n_nodes)]
#sum_x[i]  and sum_y[i] give the location of the centroid of cluster i.
#oxygen_level, distance, Eu, ram, memory,
#yet to read ram and memory from the txt file
node_relay=[False for x in xrange(n_nodes)]
for j in range(len(target_mapping)) :	#For a sensor mapping target j
	for i in range(0,n_nodes):	# For a node i
		if(  ( labels[target_mapping[j] ] == labels[i] ) and (target_mapping[j]!= i)) : #Checking if node i is in the same cluster as a sensor mapping target j # also making sure sensor is not the node
			node_relay[i] = True
			
for i in range(n_nodes):
	distance[i] = ( ( nodes[i][0] - sum_x [ labels[i] ] )**2 + ( nodes[i][1] - sum_y [ labels[i] ] )**2 ) **0.5  
	power_node[i] =  p_alpha*distance[i] + p_beta*Eu[i] + p_gamma*ram[i] + p_delta*memory[i]


node_relay1=[False for x in xrange(n_nodes)]
x_relay=[]
y_relay=[]

temp_var=[0 for x in xrange(n_clusters)]
min_power=[0 for x in xrange(n_clusters)]
for i in range (0,n_clusters):	#iterating through all the clusters
	for k in range(0,n_nodes):
		if ( (labels[k]==i) and node_relay[k]):
			temp_var[i]=k	#gives the first potential relay node whose label is i 
			min_power[i]=power_node[ temp_var[i] ]	
			break		
		else:
			min_power[i]=-1
	 	#min_dist initialised as the first instance of distance of the label i
	for j in range(n_nodes):	#iterating through all nodes  
		if(node_relay[j] and labels[j]==i):	#If node j is a potential relay and belongs to the cluster i
			if ( (power_node[j] < min_power[i]) and (min_power[i] > 0) ):
				min_power[i]=power_node[j]		
				temp_var[i]=j
				
for i in range (n_clusters):				
	if((min_power[i]>=0)):
		x_relay.append(nodes[temp_var[i]][0])
		y_relay.append(nodes[temp_var[i]][1])


plt.plot(x_node,y_node,'bs')
plt.plot(x_relay,y_relay,'g^')
plt.plot(x_sensor,y_sensor,'rs')
plt.plot(sum_x,sum_y,'gs')

print "Blue squares are nodes"
print "Red squares are sensors"
print "Red triangles are targets"
print "Green triangles are relays"
print "Number of targets are " + str(n_targets)
print "Number of sensors are " + str(len(set(target_mapping)))
print "Number of clusters are " + str(n_clusters)  
print "Number of relays are " + str(len(x_relay))
plt.show()















 
