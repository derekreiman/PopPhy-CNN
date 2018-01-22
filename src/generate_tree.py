from graph import Graph
import sys
import numpy as np

dset = sys.argv[1]

g = Graph()
g.build_graph("../data/" + dset + "/newick.txt")
ref = g.get_ref()

g.write_table("../data/" + dset + "/tree.out")

fp = open("../data/" + dset +"/label_reference.txt", 'r')
labels = fp.readline().split("['")[1].split("']")[0].split("' '")
fp.close()

scores={}

for l in labels:
    scale = 1
    if l == 'n' or l == 'leaness':
        scale = -1
    fp = open("../data/" + dset + "/" + l + "_scores.out", 'r')
    for line in fp:
        node = line.split("\t")[0]
        score = line.split("\t")[1].split("[")[1].split("]")[0].split(", ")
        v = np.median(scale * np.array(score).astype(np.float))
        if node in scores:
            	scores[node] += v
        else:
            scores[node] = v        
    fp.close()

op = open("../data/" + dset + "/tree_scores.out", 'w')
op.write("Node\tScore\n")
for s in scores:
    op.write(s + "\t" + str(scores[s]) + "\n")
op.close()

op = open("../data/" + dset + "/tree_edges.out", 'w')      
fp = open("../data/" + dset + "/tree.out")

for line in fp:
    node1 = line.split("\t")[0]
    node2 = line.split("\t")[1].split("\n")[0]
    if np.abs(scores[node1]) > 0.05 and np.abs(scores[node2]) > 0.05 and scores[node1] * scores[node2] > 0:
        v = (scores[node1] + scores[node2])/2
    else:
        v = 0
    op.write(node1 + "\t" + node2 + "\t" + str(v) + "\n")
op.close()
fp.close()	
