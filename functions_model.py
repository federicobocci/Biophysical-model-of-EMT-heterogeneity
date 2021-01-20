import numpy as np

def dummy():
	print('hi')

def subcluster_sum(cp, e, h1, h2, h3, m, s, pE1, pE2, pE3, p11, p22, p33, p12, p13, p23, two_dim=False):
	
	# fractions of E and M at boundary
	e_bd = e/(e+m)
	m_bd = m/(e+m)

	# fraction of H1, H2, H3 inside clusters
	h1_fr = h1/(h1+h2+h3)
	h2_fr = h2/(h1+h2+h3)
	h3_fr = h3/(h1+h2+h3)
	
	vec = np.zeros(s)
	
	# term coming from cluster of size s
	if two_dim:
		# number of contacts
		nE1 = 2*e_bd*h1_fr + 2*s*e*h1_fr
		nE2 = 2*e_bd*h2_fr + 2*s*e*h2_fr
		nE3 = 2*e_bd*h3_fr + 2*s*e*h3_fr
		
		n11 = 2*s*h1_fr*h1
		n22 = 2*s*h2_fr*h2
		n33 = 2*s*h3_fr*h3
		
		n12 = 2*s*h1_fr*h2 + 2*s*h2_fr*h1
		n13 = 2*s*h1_fr*h3 + 2*s*h3_fr*h1
		n23 = 2*s*h2_fr*h3 + 2*s*h3_fr*h2
		
		vec[s-1] = (s**cp)*( pE1**nE1 )*( pE2**nE2 )*( pE3**nE3 )*( p11**n11 )*( p22**n22 )*( p33**n33 )*( p12**n12 )*( p23**n23 )*( p13**n13 )
	else:
		# number of E-H1 contacts
		nE1 = 2*e_bd*h1_fr
		# number of E-H2 contacts
		nE2 = 2*e_bd*h2_fr
		# number of E-H2 contacts
		nE3 = 2*e_bd*h3_fr
		vec[s-1] = (s**cp)*( pE1**nE1 )*( pE2**nE2 )*( pE3**nE3 )
	
	
	# subcluster terms
	if s>1:
		# summation on k=[2,s]
		for p in np.arange(1,s,1):
			ms = float(p) # make it float for division
			if two_dim:
				nE1 = (2/(s-ms+1))*e_bd*h1_fr + 2*ms*e*h1_fr
				nE2 = (2/(s-ms+1))*e_bd*h2_fr + 2*ms*e*h1_fr
				nE3 = (2/(s-ms+1))*e_bd*h3_fr + 2*ms*e*h3_fr
				n11 = (2/(s-ms+1))*h1_fr*h1_fr + ((s-ms-1)/(s-m+1))*2*h1_fr*h1_fr + 2*ms*h1_fr*h1
				n22 = (2/(s-ms+1))*h2_fr*h2_fr + ((s-ms-1)/(s-m+1))*2*h2_fr*h2_fr + 2*ms*h2_fr*h2
				n33 = (2/(s-ms+1))*h3_fr*h3_fr + ((s-ms-1)/(s-m+1))*2*h3_fr*h3_fr + 2*ms*h3_fr*h3
				n12 = (2/(s-ms+1))*(2*h1_fr*h2_fr) + ((s-ms-1)/(s-m+1))*2*(2*h1_fr*h2_fr) + 2*ms*(h1_fr*h2 + h2_fr*h1)
				n13 = (2/(s-ms+1))*(2*h1_fr*h3_fr) + ((s-ms-1)/(s-m+1))*2*(2*h1_fr*h3_fr) + 2*ms*(h1_fr*h3 + h3_fr*h1)
				n23 = (2/(s-ms+1))*(2*h3_fr*h2_fr) + ((s-ms-1)/(s-m+1))*2*(2*h3_fr*h2_fr) + 2*ms*(h3_fr*h2 + h2_fr*h3)
				vec[p-1] = (ms**cp)*( pE1**nE1 )*( pE2**nE2 )*( pE3**nE3 )*( p11**n11 )*( p22**n22 )*( p33**n33 )*( p12**n12 )*( p23**n23 )*( p13**n13 )
			else:
				nE1 = (2/(s-m+1))*e_bd*h1_fr
				nE2 = (2/(s-m+1))*e_bd*h2_fr
				nE3 = (2/(s-m+1))*e_bd*h3_fr
				n11 = (2/(s-ms+1))*h1_fr*h1_fr + ((s-ms-1)/(s-m+1))*2*h1_fr*h1_fr
				n22 = (2/(s-ms+1))*h2_fr*h2_fr + ((s-ms-1)/(s-m+1))*2*h2_fr*h2_fr
				n33 = (2/(s-ms+1))*h3_fr*h3_fr + ((s-ms-1)/(s-m+1))*2*h3_fr*h3_fr
				n12 = (2/(s-ms+1))*(2*h1_fr*h2_fr) + ((s-ms-1)/(s-m+1))*2*(2*h1_fr*h2_fr)
				n13 = (2/(s-ms+1))*(2*h1_fr*h3_fr) + ((s-ms-1)/(s-m+1))*2*(2*h1_fr*h3_fr)
				n23 = (2/(s-ms+1))*(2*h2_fr*h3_fr) + ((s-ms-1)/(s-m+1))*2*(2*h2_fr*h3_fr)
				vec[p-1] = (ms**cp)*( pE1**nE1 )*( pE2**nE2 )*( pE3**nE3 )*( p11**n11 )*( p22**n22 )*( p33**n33 )*( p12**n12 )*( p23**n23 )*( p13**n13 )
	return vec



def cluster_sum(cp, e, h1, h2, h3, m, pE1, pE2, pE3, p11, p22, p33, p12, p13, p23, two_dim=False):
	h = h1+h2+h3
	if h==0:
		return 0., np.array([])
	else:
		# threshold to stop summation (found by trial-and-error)
		epsilon = 0.0001
		c = 0.
		s = 1
		# vector to store contribution to escape rate by clusters of size s
		clst_list = []
		while ( s>0 ):
			# each iteration evaluates clusters of size s
			# subcluster_sum evaluates all contributions by subclusters if size m<s
			sub_cont = subcluster_sum(cp, e, h1, h2, h3, m, s, pE1, pE2, pE3, p11, p22, p33, p12, p13, p23, two_dim=two_dim)
			inc = (h**s)*((1-h)**2)*(2/(s*(s+1)))*np.sum( sub_cont )
			clst_list.append( (h**s)*((1-h)**2)*(2/(s*(s+1)))*( sub_cont ) )
			c = c + inc
			s = s + 1
			# when exceeding cutoff, stop summing
			if inc/c < epsilon:
				break
		return c, clst_list

def solve_model(c, k, pE1, pE2, pE3, p11, p22, p33, p12, p13, p23, two_dim=False):
	'''
	returns fractions of cells vs time
	c: cooperation parameter
	k: emt rate
	pij: set of probabilities to break adhesion bonds
	two_dim: if True, solve using the effective-2D approximation (always True)
	'''
	dt = 0.0025 # dt found by trila-and-error, small enough to guarantee convergence
	T = 10*max(1./k, 1.0) # ensure convergence to steady state

	# define arrays to store e, h1, h2, h3, m cell fraction and time
	tm = np.arange(0, T + dt, dt)
	npoints = int(T / dt)
	e = np.zeros(npoints+1)
	h1 = np.zeros(npoints+1)
	h2 = np.zeros(npoints+1)
	h3 = np.zeros(npoints+1)
	m = np.zeros(npoints+1)

	# initial condition with all epithelial cells
	e[0] = 1.

	for i in range( npoints ):
		# relative fractions of hybrid cells in states h1, h2, h3
		
		if (h1[i]!=0 and h2[i]!=0) and (h3[i]!=0):
			h1_fr = (h1[i]/(h1[i]+h2[i]+h3[i]))
			h2_fr = (h2[i]/(h1[i]+h2[i]+h3[i]))
			h3_fr = (h3[i]/(h1[i]+h2[i]+h3[i]))
		else:
			h1_fr = 0.
			h2_fr = 0.
			h3_fr = 0.
		# escape term for hybrid cells h1, h2, h3
		sum, list = cluster_sum(c, e[i], h1[i], h2[i], h3[i], m[i], pE1, pE2, pE3, p11, p22, p33, p12, p13, p23, two_dim=two_dim)
		
		e[i+1] = e[i] + dt*( -k*e[i] + sum + m[i] )
		h1[i+1] = h1[i] + dt*( +k*(e[i]-h1[i]) - h1_fr*sum )
		h2[i+1] = h2[i] + dt*( +k*(h1[i]-h2[i]) - h2_fr*sum )
		h3[i+1] = h3[i] + dt*( +k*(h2[i]-h3[i]) - h3_fr*sum )
		m[i+1] = m[i] + dt*( +k*h3[i] - m[i] )
	
	return tm, e, h1, h2, h3, m



def cluster_sum_ss(num, cp, e, h1, h2, h3, m, pE1, pE2, pE3, p11, p22, p33, p12, p13, p23, two_dim=False):
	'''
	returns the flux of cluster and considers only clusters up to size num
	only difference between this function and cluster_sum is that here the max cluster size (num) is fixed
	'''
	h = h1+h2+h3
	if h==0:
		return 0., np.array([])
	else:
		epsilon = 0.0001
		c = 0.
		# vector to store contribution to escape rate by clusters if size s
		clst_list = []
		for s in np.arange(1,num+1,1):
			sub_cont = subcluster_sum(cp, e, h1, h2, h3, m, s, pE1, pE2, pE3, p11, p22, p33, p12, p13, p23, two_dim=two_dim)
			inc = ( 2*(h**s)*((1-h)**2)/(s*(s+1)) )*np.sum( sub_cont )
			clst_list.append( ( 2*(h**s)*((1-h)**2)/(s*(s+1)) )*( sub_cont ) )
			c = c + inc

		return c, clst_list


def find_dist(clst_list):
	maxs = len(clst_list)
	rates = np.zeros((maxs, maxs))
	dist = np.zeros(maxs)

	for i in range(maxs):
		for j in range(maxs):
			if j<clst_list[i].size:
				rates[i][j] = clst_list[i][j]
			else:
				rates[i][j] = 0.

	for i in range(maxs):
		dist[i] = np.sum( rates[:,i] )

	return dist













