###################################
#!/usr/bin/env python
#coding=utf-8
# SPRINT-Str.py
# Ghazaleh Taherzadeh
###################################
"""predict Peptide-binding residues for a new chain"""
import sys
import string
import math
import os
import numpy
import scipy
import scipy.io
import pandas as pd
import numpy as np
import cPickle
import pickle

   
def compute_entropy(dis_list):
    """compute shannon entropy for a distribution.
    base = len(dis_list) is the base of log function 
    to make entropy between 0 and 1."""
    
    if sum(dis_list) == 0:
        return 0.0
    prob_list = map(lambda x:(x+0.0)/sum(dis_list),dis_list)
    ent = 0.0
    for prob in prob_list:
        if prob != 0:
            ent -= prob*math.log(prob,len(dis_list))
    return ent

def compute_ss_content(ss_seq_win):
    """compute ss content in a window."""
    con_C = con_H = con_E = 0
    for ss in ss_seq_win:
        if ss == 'C':
            con_C += 1
        elif ss == 'H':
            con_H += 1
        elif ss == 'E':
            con_E += 1
        else:
            print('X')
             
    act_len = con_C+con_H+con_E+0.0
    return ['%.3f'%(con_C/act_len),'%.3f'%(con_H/act_len),'%.3f'%(con_E/act_len)]

def ss_binary(ss_type):
    binary = []
    for ss in ss_type:
	if ss == 'C':
	   binary.append('1.000')
           binary.append('0.000')
           binary.append('0.000')
	elif ss == 'H':
	   binary.append('0.000')
	   binary.append('1.000')
	   binary.append('0.000')
	elif ss == 'E':
	   binary.append('0.000')
	   binary.append('0.000')
	   binary.append('1.000')
	else:
	   binary.append('0.000')
	   binary.append('0.000')
	   binary.append('0.000')
    return binary

def ss_to_num(sin_ss):
    """C->0,H->1,E->2,'$'->-1"""
    if sin_ss == 'C':
        return 0
    elif sin_ss == 'H':
        return 1
    elif sin_ss == 'E':
        return 2
    else:
        return -1

def seg_bound(s_win):
    """Two boundaries of a segment"""
    c_ss = s_win[len(s_win)/2]
    l_len = r_len = 0
    i = len(s_win)/2 - 1
    while i >= 0:
        if s_win[i] != c_ss:
            break
        l_len += 1
        i -= 1
    i = len(s_win)/2 + 1
    while i < len(s_win):
        if s_win[i] != c_ss:
            break
        r_len += 1
        i += 1
    return (l_len,r_len)

ext_state = {'A':115.0,'D':150.0,'C':135.0,'E':190.0,'F':210.0,\
             'G':75.0,'H':195.0,'I':175.0,'K':200.0,'L':170.0,\
             'M':185.0,'N':160.0,'P':145.0,'Q':180.0,'R':225.0,\
             'S':115.0,'T':140.0,'V':155.0,'W':255.0,'Y':230.0,'X':100000}

ext_pdb_residue = {'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E', 'PHE':'F',\
               'GLY':'G', 'HIS':'H', 'ILE':'I', 'LYS':'K', 'LEU':'L',\
			   'MET':'M', 'ASN':'N', 'PRO':'P', 'GLN':'Q', 'ARG':'R',\
			   'SER':'S', 'THR':'T', 'VAL':'V', 'TRP':'W', 'TYR':'Y'}
#pid = r'1a81A' ./SPRINT.py 1a81A
pid = sys.argv[1] 
G1_win = 4#HSE
G2_win = 5#PSSM
G3_win = 2#SS
G4_win = 4#ASA 
base_path = 'path_to_your_pdb_dir'
res_path = base_path+'files/'
dssp_path = base_path+'/dssp_result/'
pssm_path = base_path+'/pssm_pssm/'
HSE_path = base_path+'/HSE/'
info_path = base_path+'files/'
fea_path = base_path+'files/'
error_file = base_path+'error.log'
#/*************************************************/
# build info file
pdb_residue_temp = []
pdb_residue = []
all_coord = []
all_X = []
all_Y = []
all_Z = []
b_factor = []
pdbfile = []
all_residue_number = []
residue_number = []
#change the path
with open('your_pdb_directory'+pid+'.pdb') as pdb_file:
	for line in pdb_file:
		if line[:4] == 'ATOM' or line[:6] == "HETATM" and line:
			temp_pdbfile = ('%s%5.2f'%(line[:55].rstrip(),1.00))
			pdbfile.append(temp_pdbfile)
			all_residue_number.append(line[22:26])
			if line[12:16] == ' CA ':
				residue_number.append(line[22:26])
				pdb_residue_temp.append(line[17:20])
				all_coord.append([line[30:38], line[38:46], line[46:54]])
				all_X.append(line[30:38])
				all_Y.append(line[38:46])
				all_Z.append(line[46:54])
				b_factor.append(line[57:60])
all_residue_number = [x.replace(' ', '') for x in all_residue_number]
for i in xrange(len(pdb_residue_temp)):
	pdb_residue.append(ext_pdb_residue[pdb_residue_temp[i]])
	

fin = file(HSE_path+pid+'.pdb.txt','r')
aa = fin.readlines()[1:] 
fin.close()
HSE_res = map(lambda x:x.split()[3],aa)
act_CN = map(lambda x:x.split()[4],aa)
act_HSE_UP = map(lambda x:x.split()[5],aa)
act_HSE_DN = map(lambda x:x.split()[6],aa)
act_CN = [string.atof(x.strip(' ')) for x in act_CN]
min_cn = min(act_CN)
max_cn = max(act_CN)
CN = [((q - min_cn)/(max_cn-min_cn)) for q in act_CN]
act_HSE_UP = [string.atof(x.strip(' ')) for x in act_HSE_UP]
min_up = min(act_HSE_UP)
max_up = max(act_HSE_UP)
HSE_UP = [((q - min_up)/(max_up-min_up)) for q in act_HSE_UP]
act_HSE_DN = [string.atof(x.strip(' ')) for x in act_HSE_DN]
min_dn = min(act_HSE_DN)
max_dn = max(act_HSE_DN)
HSE_DN = [((q - min_dn)/(max_dn-min_dn)) for q in act_HSE_DN]

fin = file(pssm_path+pid+'.pssm','r')
pssm = fin.readlines()
fin.close()
if len(pssm[-6].split()) != 0 or pssm[3].split()[0] != '1': 
    print 'error on reading pssm, line -6 is not a spare line;\
     or line 3 is not the first line'
    sys.exit(1)
pssm = pssm[3:-6]
pssm_res = map(lambda x:x.split()[1],pssm)

fin = file(dssp_path+pid+'.pdb.txt','r')
dssp = fin.readlines()[1:] 
fin.close()

cnt_dssp = 0
for line in dssp:
	cnt_dssp += 1
	if '#' in line:
		strt_pnt = cnt_dssp
dssp = dssp[strt_pnt:]




ss = []
asa = []
dssp_res = []
cnt = 0
for i in xrange(len(dssp)):
    ss.append (dssp[i][14:18])
    asa.append (dssp[i][34:38])
    dssp_res.append(dssp[i][12:14])
dssp_res = [x.strip(' ') for x in dssp_res]
matching = [s for s, char in enumerate(dssp_res) if char == '!']
if (len(matching)) <> 0:
	for i in xrange(len(matching)):
		index_matching = matching[i]
		if dssp_res[index_matching+1] == pdb_residue[index_matching]:# Check if dssp_res == '!', the residue should be removed
			dssp_res.remove(dssp_res[index_matching-i])
			ss.remove(ss[index_matching-i])
			asa.remove(asa[index_matching-i])
		elif dssp_res[index_matching+1] == pdb_residue[index_matching+1]: # Check if dssp_res == '!', the residue should change to M
			dssp_res[index_matching-i] = 'M'
		
if not ''.join(pssm_res) == ''.join(dssp_res) == ''.join(HSE_res):
    print 'Sequence inconsistent!'
    print 'pssm: ',''.join(pssm_res)
    print 'dssp: ',''.join(dssp_res)
    print ' HSE: ',''.join(HSE_res)
    exit(1)
ss = [x.strip(' ') for x in ss]
asa = [string.atof(x.strip(' ')) for x in asa]
dssp_ss = []
for i in xrange(len(ss)):
	if ss[i] in 'CST':
		dssp_ss.append('C')
	elif ss[i] in 'HGI':
		dssp_ss.append('H')
	elif ss[i] in 'EBb':
		dssp_ss.append('E')
	elif ss[i] == '':
		dssp_ss.append('C')

dssp_rsa = []
for i in xrange(len(asa)):
	temp = round(asa[i]/ext_state[dssp_res[i]],3)
	dssp_rsa.append(temp)


fout = file(info_path+pid+'.info','w')
fout.write('>%s\n' %pid)
fastaseq = ''.join(pssm_res)
pos = 0
for i in xrange(len(fastaseq)):
    res = fastaseq[i]
    fout.write('%5d%5s%5s'%(i+1,res,res))
    if pssm[pos].split()[1] == res:
        for p_e in pssm[pos].split()[2:22]:
            fout.write(':%2s' %p_e)
        for p_e in pssm[pos].split()[22:42]:
            fout.write(':%3s' %p_e)
        fout.write(':%5s' %pssm[pos].split()[42])
    else:
        print 'Error reading pssm file!'
        flog = file(error_file,'a')
        flog.write(pid+': error on writing pssm, %s:%s\n' \
        %(pssm[pos].split()[1],res))
        flog.close()
        sys.exit(1)
    if dssp_res[pos] == res:
        fout.write(':%s' %dssp_ss[pos])
    else:
        print 'Error reading ss file!'
        flog = file(error_file,'a')
        flog.write(pid+': error on writing ss, %s:%s\n' %(ss_res[pos],res))
        flog.close()
        sys.exit(1)
    if dssp_res[pos] == res:
	fout.write(':%s' %dssp_rsa[pos])
    else:
        print 'Error reading rsa file!'
        flog = file(error_file,'a')
        flog.write(pid+': error on writing rsa, %s:%s\n' %(rsa_res[pos],res))
        flog.close()
        sys.exit(1)
    pos += 1
    fout.write('\n')
fout.close()

#/*************************************************/
fin = file(info_path+'%s.info'%pid,'r')
info = fin.readlines()[1:]
fin.close()
output = file(fea_path+'%s.fea'%pid,'w')
seq_len = len(info)
out_list = []
for i in xrange(len(info)):
    out_list.append([])
pssm = map(lambda x:map(lambda y:'%7.5f' %(1/(1+math.pow(math.e,-string.atoi(y)))),x.split(':')[1:21]),info)
pssm_t = []
for i in xrange(20):
    pssm_t.append('%7.5f' %(1/(1+math.e**0)))
for i in xrange(G2_win):
    pssm.insert(0,pssm_t)
    pssm.append(pssm_t)
for i in xrange(G2_win,seq_len+G2_win):
    for j in xrange(i-G2_win,i+G2_win+1):
        out_list[i-G2_win].append(','.join(pssm[j]))
wop = ['%7.5f' %compute_entropy(z) for z in map(lambda x:map(lambda y:string.atoi(y),x.split(':')[21:41]),info)]
for i in xrange(G2_win):
    wop.insert(0,'%7.5f' %(0))
    wop.append('%7.5f' %(0))
for i in xrange(G2_win,seq_len+G2_win):
    for j in xrange(i-G2_win,i+G2_win+1):
        out_list[i-G2_win].append(wop[j])
ss_seq = map(lambda x:x.split(':')[42],info)
for i in xrange(G3_win):
    ss_seq.insert(0,'$')
    ss_seq.append('$')  
for i in xrange(G3_win,seq_len+G3_win):
    out_list[i-G3_win].append(','.join(compute_ss_content(ss_seq[i-G3_win:i+G3_win+1])))
for i in xrange(G3_win,seq_len+G3_win):
    out_list[i-G3_win].append(','.join(ss_binary(ss_seq[i-G3_win:i+G3_win+1])))
temp = []    
for i in xrange(G3_win,seq_len+G3_win):
    [l_b,r_b] = seg_bound(ss_seq[i-G3_win:i+G3_win+1])
    for j in xrange(3):#0-C,1-H,2-E
	if j == ss_to_num(ss_seq[i]):
            out_list[i-G3_win].append('%.3f'%((l_b+r_b+1.0)/(2*G3_win+1)))
	    temp.append('%.3f'%((l_b+r_b+1.0)/(2*G3_win+1)))
        else:
            out_list[i-G3_win].append('0.000')
	    temp.append('0.000')
    for j in xrange(3):
        if j == ss_to_num(ss_seq[i]):
            out_list[i-G3_win].append('%.3f'%((min(l_b,r_b)+0.0)/G3_win))
	    temp.append('%.3f'%((min(l_b,r_b)+0.0)/G3_win))
        else:
            out_list[i-G3_win].append('0.000')
	    temp.append('0.000')
    for j in xrange(3):
        if j == ss_to_num(ss_seq[i]):
            out_list[i-G3_win].append('%.3f'%((max(l_b,r_b)+0.0)/G3_win))
	    temp.append('%.3f'%((max(l_b,r_b)+0.0)/G3_win))
        else:
            out_list[i-G3_win].append('0.000')
	    temp.append('0.000')
    
rsa = map(lambda x:x.split(':')[43].split()[0],info)
for i in xrange(G4_win):
    rsa.insert(0,'1.000')
    rsa.append('1.000')
for i in xrange(G4_win,seq_len+G4_win):
    for j in xrange(i-G4_win,i+G4_win+1):
        out_list[i-G4_win].append(rsa[j])

rsa = [string.atof(x.split(':')[43]) for x in info]
for i in xrange(G4_win):
    rsa.insert(0,1.0)
    rsa.append(1.0)
for i in xrange(G4_win,seq_len+G4_win):
    for j in xrange(1,G4_win+1):
        out_list[i-G4_win].append('%.4f'%(sum(rsa[i-j:i+j+1])/(2*j+1)))
for i in xrange(G1_win):
    CN.insert(0,0.000)
    CN.append(0.000)
for i in xrange(G1_win,seq_len+G1_win):
    for j in xrange(i-G1_win,i+G1_win+1):
	out_list[i-G1_win].append('%.4f'%(CN[j]))
for i in xrange(G1_win):
    HSE_UP.insert(0,0.000)
    HSE_UP.append(0.000)
for i in xrange(G1_win,seq_len+G1_win):
    for j in xrange(i-G1_win,i+G1_win+1):
	out_list[i-G1_win].append('%.4f'%(HSE_UP[j]))
for i in xrange(G1_win):
    HSE_DN.insert(0,0.000)
    HSE_DN.append(0.000)
for i in xrange(G1_win,seq_len+G1_win):
    for j in xrange(i-G1_win,i+G1_win+1):
	out_list[i-G1_win].append('%.4f'%(HSE_DN[j]))
for i in xrange(seq_len):
    out_list[i].append('%.4f'%(0.000))

for i in xrange(len(out_list)):
    output.write(','.join(out_list[i])+'\n')
output.close()

#/*************************************************/
from sklearn.ensemble import RandomForestClassifier


testdata = np.loadtxt(base_path+'files/'+pid+'.fea', delimiter=",")
numcols = len(testdata[0])
test_sample = testdata[:,0:numcols-1]
test_label = testdata[:,numcols-1]


with open(base_path+'rf_model.pkl', 'rb') as f:
    clf = cPickle.load(f)
probability = clf.predict_proba(test_sample)
prob_score = probability[:,1]
predicted_prob = prob_score
#normalize probability
for i in xrange(len(prob_score)):
	if prob_score[i] < 0.2:
		oldMin = 0
		oldMax = 0.1999
		newMin = 1
		newMax = 4
		newRange = newMax - newMin
		oldRange = oldMax - oldMin
		predicted_prob[i] = round(((prob_score[i]- oldMin) * newRange / oldRange) + newMin)
	elif prob_score[i] >= 0.2:
		oldMin = 0.2
		oldMax = 1
		newMin = 5
		newMax = 9
		newRange = newMax - newMin
		oldRange = oldMax - oldMin
		predicted_prob[i] = round(((prob_score[i]- oldMin) * newRange / oldRange) + newMin)
predicted_prob = predicted_prob.astype(int)
print type(predicted_prob)
raw_input('probability')
#

residue_prediction = [i for i in range(len(prob_score)) if prob_score[i] >= 0.2]


#/*************************************************/
from sklearn.cluster import DBSCAN
from numpy import array

nigh = 30
th = [0.2, 0.17, 0.15, 0.13, 0.1, 0.09, 0.07, 0.05, 0.03, 0.01]
next = 0 
accept = 0
while (accept == 0) and (next <= 8):
	numofresidue = 0
	select = 0  
	predicted_residues = [i for i in range(len(prob_score)) if prob_score[i] >= th[next]]
	if len(predicted_residues) == 0:
		next += 1
	else:
		next += 1
		coord = []
		X = []
		Y = []
		Z = []
		for i in range(len(predicted_residues)):
			coord.append(all_coord[predicted_residues[i]])
			X.append(all_X[predicted_residues[i]])
			Y.append(all_Y[predicted_residues[i]])
			Z.append(all_Z[predicted_residues[i]])

		db_r1 = DBSCAN(eps=7, min_samples=1).fit(coord)
		labels_r1 = db_r1.labels_
		n_clusters_r1 = len(set(labels_r1)) - (1 if -1 in labels_r1 else 0)
		count_r1 = np.count_nonzero(labels_r1 == -1)
		db_r2 = DBSCAN(eps=7, min_samples=2).fit(coord)
		labels_r2 = db_r2.labels_
		n_clusters_r2 = len(set(labels_r2)) - (1 if -1 in labels_r2 else 0)
		count_r2 = np.count_nonzero(labels_r2 == -1)
		db_r3 = DBSCAN(eps=7, min_samples=3).fit(coord)
		labels_r3 = db_r3.labels_
		n_clusters_r3 = len(set(labels_r3)) - (1 if -1 in labels_r3 else 0)
		count_r3 = np.count_nonzero(labels_r3 == -1)
		db_r4 = DBSCAN(eps=7, min_samples=4).fit(coord)
		labels_r4 = db_r4.labels_
		n_clusters_r4 = len(set(labels_r4)) - (1 if -1 in labels_r4 else 0)
		count_r4 = np.count_nonzero(labels_r4 == -1)
		db_r5 = DBSCAN(eps=7, min_samples=5).fit(coord)
		labels_r5 = db_r5.labels_
		n_clusters_r5 = len(set(labels_r5)) - (1 if -1 in labels_r5 else 0)
		count_r5 = np.count_nonzero(labels_r5 == -1)
		db_r6 = DBSCAN(eps=7, min_samples=6).fit(coord)
		labels_r6 = db_r6.labels_
		n_clusters_r6 = len(set(labels_r6)) - (1 if -1 in labels_r6 else 0)
		count_r6 = np.count_nonzero(labels_r6 == -1)
		db_r7 = DBSCAN(eps=7, min_samples=7).fit(coord)
		labels_r7 = db_r7.labels_
		n_clusters_r7 = len(set(labels_r7)) - (1 if -1 in labels_r7 else 0)
		count_r7 = np.count_nonzero(labels_r7 == -1)
		db_r8 = DBSCAN(eps=7, min_samples=8).fit(coord)
		labels_r8 = db_r8.labels_
		n_clusters_r8 = len(set(labels_r8)) - (1 if -1 in labels_r8 else 0)
		count_r8 = np.count_nonzero(labels_r8 == -1)
		db_r9 = DBSCAN(eps=7, min_samples=9).fit(coord)
		labels_r9 = db_r9.labels_
		n_clusters_r9 = len(set(labels_r9)) - (1 if -1 in labels_r9 else 0)
		count_r9 = np.count_nonzero(labels_r9 == -1)
		db_1= DBSCAN(min_samples=1).fit(coord)
		labels_1 = db_1.labels_
		n_clusters_1 = len(set(labels_1)) - (1 if -1 in labels_1 else 0)
		count_1 = np.count_nonzero(labels_1 == -1)
		db_2 = DBSCAN(min_samples=2).fit(coord)
		labels_2 = db_2.labels_
		n_clusters_2 = len(set(labels_2)) - (1 if -1 in labels_2 else 0)
		count_2 = np.count_nonzero(labels_2 == -1)
		db_3 = DBSCAN(min_samples=3).fit(coord)
		labels_3 = db_3.labels_
		n_clusters_3 = len(set(labels_3)) - (1 if -1 in labels_3 else 0)
		count_3 = np.count_nonzero(labels_3 == -1)
		db_4 = DBSCAN(min_samples=4).fit(coord)
		labels_4 = db_4.labels_
		n_clusters_4 = len(set(labels_4)) - (1 if -1 in labels_4 else 0)
		count_4 = np.count_nonzero(labels_4 == -1)
		db_5 = DBSCAN(min_samples=5).fit(coord)
		labels_5 = db_5.labels_
		n_clusters_5 = len(set(labels_5)) - (1 if -1 in labels_5 else 0)
		count_5 = np.count_nonzero(labels_5 == -1)
		db_6 = DBSCAN(min_samples=6).fit(coord)
		labels_6 = db_6.labels_
		n_clusters_6 = len(set(labels_6)) - (1 if -1 in labels_6 else 0)
		count_6 = np.count_nonzero(labels_6 == -1)
		db_7 = DBSCAN(min_samples=7).fit(coord)
		labels_7 = db_7.labels_
		n_clusters_7 = len(set(labels_7)) - (1 if -1 in labels_7 else 0)
		count_7 = np.count_nonzero(labels_7 == -1)
		db_8 = DBSCAN(min_samples=8).fit(coord)
		labels_8 = db_8.labels_
		n_clusters_8 = len(set(labels_8)) - (1 if -1 in labels_8 else 0)
		count_8 = np.count_nonzero(labels_8 == -1)
		db_9 = DBSCAN(min_samples=9).fit(coord)
		labels_9 = db_9.labels_
		n_clusters_9 = len(set(labels_9)) - (1 if -1 in labels_9 else 0)
		count_9 = np.count_nonzero(labels_9 == -1)
	
	
		all_r = [count_r1, count_r2, count_r3, count_r4, count_r5, count_r6, count_r7, count_r8, count_r9]
		labels_r = [labels_r1, labels_r2, labels_r3, labels_r4, labels_r5, labels_r6, labels_r7, labels_r8, labels_r9]
		countnoise_r = all_r.index(min(all_r))
		all = [count_1, count_2, count_3, count_4, count_5, count_6, count_7, count_8, count_9]
		labels = [labels_1, labels_2, labels_3, labels_4, labels_5, labels_6, labels_7, labels_8, labels_9]
		countnoise = all.index(min(all))
	     
		selected_coord = []
		selected_X = []
		selected_Y = []
		selected_Z = []
		if all_r[countnoise_r] <= all[countnoise]:
			temp_label = labels_r[countnoise_r]
			temp_idx = numpy.where(labels_r[countnoise_r] != -1)
		else:
			temp_label = labels[countnoise]
			temp_idx = numpy.where(labels[countnoise] != -1)
	
		selected_label = temp_label[temp_label != -1]
		idx = temp_idx[0].tolist()
		size_largestclust =  np.bincount(selected_label)
		largest_clust_no = np.argmax(np.bincount(selected_label))
		largest_clust_ind_temp = np.where(selected_label==largest_clust_no)
		largest_clust_ind = largest_clust_ind_temp[0].tolist()
		final_index = []
		if max(np.bincount(selected_label)) >= 15 :
			if max(np.bincount(selected_label)) > 30:
				for i in range(len(largest_clust_ind)):
					selected_coord.append(coord[largest_clust_ind[i]])
					selected_X.append(X[largest_clust_ind[i]])
					selected_Y.append(Y[largest_clust_ind[i]])
					selected_Z.append(Z[largest_clust_ind[i]])
				X_temp = list(map(float, selected_X))
				Y_temp = list(map(float, selected_Y))
				Z_temp = list(map(float, selected_Z))
				X_mean =  reduce(lambda x, y: x + y, X_temp) / len(X_temp)
				Y_mean =  reduce(lambda x, y: x + y, Y_temp) / len(Y_temp)
				Z_mean =  reduce(lambda x, y: x + y, Z_temp) / len(Z_temp)
				distance = []
				for dis in range(len(selected_X)):
					distance.append(math.sqrt(((X_mean-float(selected_X[dis]))**2) + ((Y_mean-float(selected_Y[dis]))**2) + ((Z_mean-float(selected_Z[dis]))**2)))
					sorted_index = sorted(range(len(distance)), key=lambda k: distance[k])
				final_index_temp = sorted_index[0:30]
				for i in range(len(final_index_temp)):
					final_index.append(predicted_residues[idx[largest_clust_ind[final_index_temp[i]]]])
			else:
				for i in range(len(largest_clust_ind)):
					final_index.append(predicted_residues[idx[largest_clust_ind[i]]])
			accept = 1
#---------------------------------------------------
Predicted_binding_residues = [0] * len(pdb_residue)
Predicted_binding_sites = [0] * len(pdb_residue)
for i in xrange(len(final_index)):
	Predicted_binding_sites[final_index[i]] = 1
outfile = file(base_path+'files/'+pid+'.out','w')
outfile.write('Protein sequence  :'+'\n')
outfile.write("".join(pdb_residue)+'\n')
outfile.write('Predicted Residue :'+'\n')
outfile.write("".join(map(str,predicted_prob))+'\n')
outfile.write('Predicted site    :'+'\n')
outfile.write("".join(map(str,Predicted_binding_sites))+'\n')
#/*************************************************/
pdb_predicted_index = []
for i in xrange(len(final_index)):
	pdb_predicted_index.append(residue_number[final_index[i]])

pdb_predicted_index = [x.replace(' ', '') for x in pdb_predicted_index]
for k in xrange(len(pdb_predicted_index)):
	selected_indices = [i for i,x in enumerate(all_residue_number) if x == pdb_predicted_index[k]]
	for j in xrange(len(selected_indices)):
		temp_line = pdbfile[selected_indices[j]]
		temp_line = temp_line.replace(temp_line[55],"9")
		pdbfile[selected_indices[j]] = temp_line
		
fileout = file(base_path+'files/'+pid+'.pdb','w')
fileout.write('\n'.join(pdbfile))
