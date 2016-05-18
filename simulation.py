from random import randint
import matplotlib.pyplot as plt
import numpy as np
#from matplotlib import style
from sklearn.cluster import KMeans

print "please start entering"
n_nodes=200
n_targets=30
r_sensing=20
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
	nodes[i][0]=randint(0,100)
	nodes[i][1]=randint(0,100)
	
for i in range (0,n_targets):
	targets[i][0]=randint(0,100)
	targets[i][1]=randint(0,100)
	
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
#plt.plot(x_target, y_target,'bs',x_node, y_node,'g^')



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

for i in range(0,n_targets):
	circle=plt.Circle((x_sensor[i],y_sensor[i]),r_sensing,color='r')
	fig = plt.gcf()
	fig.gca().add_artist(circle)

plt.plot(x_sensor,y_sensor,'bs',x_target,y_target,'g^')
	


kmeans=KMeans(n_clusters = n_nodes-len(set(target_mapping)) )
kmeans.fit(nodes)
print "Fitting done"
centroids=kmeans.cluster_centers_
labels = kmeans.labels_

colors = ["g.","r."]

for i in range (n_nodes-len(set(target_mapping))):
	plt.plot(nodes[i][0],nodes[i][1],markersize=10)
	
	

plt.show()
	



	
