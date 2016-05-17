from random import randint
import matplotlib.pyplot as plt
n_nodes=int(raw_input())
n_targets=int(raw_input())
r_sensing=int(raw_input())

TUinitial=0
TU=[[0 for y in xrange(1)]for x in xrange(n_nodes)]

TargetSU=[[0 for y in xrange(1)]for x in xrange(n_nodes)]
targetval=n_targets

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


xs = [x[0] for x in targets]
ys = [x[1] for x in targets]
x1s = [x[0] for x in nodes]
y1s = [x[1] for x in nodes]
plt.plot(xs, ys,'bs',x1s, y1s,'g^')
plt.show()

#after this comment we need to be assured that the points are covered so we can use the bada_mat
 					
#the sum of any row in the bada_mat gives us the value of targetSu required in the algorithm 

#while targetval>0 :
#	for i in range(0,n_nodes):
#		for j in range(0,n_targets):
#			TargetSU[i][0]=TargetSU[i][0]+bada_mat[i][j]
#	for i in range(0,n_nodes):
		#insert formula for TU[i] here interms of residual energy alpha beta TargetSU[i][0] TUinitial 
#after getting the TU array we get the minimum of the array 


	
			
 
