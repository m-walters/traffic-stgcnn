
# coding: utf-8

# In[ ]:


import numpy as np
import graphtools as gt
from importlib import reload
import pandas as pd
import matplotlib as plt

get_ipython().magic(u'matplotlib notebook')


# In[ ]:


# Hyperparams
vfname = "veldata/secondring_t1.0v1.0l10"
info = {}
infofname = vfname+".info"
print("Creating info dict")
info = gt.get_info_dict(infofname)
for key,val in sorted(info.items()):
    print(key,"\t",val)


# In[ ]:


reload(gt)


# In[ ]:


# Gather nodes
xmin,xmax = info["xmin"]/gt.long2km, info["xmax"]/gt.long2km
ymin,ymax = info["ymin"]/gt.lat2km, info["ymax"]/gt.lat2km
region_gsi = [xmin,xmax,ymin,ymax]
# A combines edges and connections to their idx in senders/receivers
nodes, edges = gt.generate_nodes(region=region_gsi, mindist=0.5, maxdist=2., maxnbr=8)

n_nodes, n_edges = len(nodes.index), len(edges.index)
print("Number of nodes", n_nodes)
print("Number of edges", n_edges)


# In[ ]:


# 5000 pts takes about 30 seconds
vdf = gt.get_veldf(vfname,nodedf=nodes,days=[],nTG=info["nTG"])
vdf.drop_duplicates(inplace=True)
print(len(vdf.index))


# In[ ]:


# five files
node_fname = "nn_inputs/node_features"
edge_fname = "nn_inputs/edge_features"
send_fname = "nn_inputs/senders"
receive_fname = "nn_inputs/receivers"
glbl_fname = "nn_inputs/glbls"


# In[ ]:


# Node features:   ncars, v_avg, v_std
# Edge features:   ncars, v_avg, v_std, pol (polarity, towards/away)
# For edge features, perhaps capture cars that are within +/- pi/4 radians of the edge's line
# Don't forget about the cars going in the opposite direction too
# Picture a bowtie centered on the edge's "road line"
# For averaging, do a weighted sum based on dist2node

nodes["ncar"] = 0
nodes["v_avg"] = 0.
nodes["v_std"] = 0.

edges["ncar_out"] = 0
edges["ncar_in"] = 0
edges["v_avg_out"] = 0.
edges["v_avg_in"] = 0.
edges["v_std_out"] = 0.
edges["v_std_in"] = 0.

nsnap = 7*info["nTG"]
node_feat_arr = np.zeros(shape=(nsnap, n_nodes, 3), dtype=np.float)
edge_feat_arr = np.zeros(shape=(nsnap, n_edges, 6), dtype=np.float)
send_arr = edges[["sender"]].to_numpy(dtype=np.float).reshape((n_edges))
rece_arr = edges[["receiver"]].to_numpy(dtype=np.float).reshape((n_edges))
glbl_arr = np.zeros(shape=(nsnap,2), dtype=np.float)

i_edge = 0
for day in range(7):
    for tg in range(info["nTG"]):
# for day in [0]:
#     for tg in [1]:
        # Get velstats for this day, tg
        vdf_ = vdf[(vdf['day']==day) & (vdf['tg']==tg)]
        for idx, node in nodes.iterrows():
            # Give this subset of vels, calc stats for each node
            vels = vdf_[vdf_['nodeID'] == idx]
            nodes.at[idx,'ncar'] = len(vels.index)
            if len(vels.index) == 0:
                continue
            nodes.at[idx,'v_avg'] = vels.mean(axis=0)['v']
            if len(vels.index) > 1:
                nodes.at[idx,'v_std'] = vels.std(axis=0)['v']
        
            # Iterate over this nodes edges, adding vel stats as necessary
            edges_ = edges[edges["sender"] == idx]
            for eidx, e in edges_.iterrows():
                v_out, v_in = [], []
                for iv, v in vels.iterrows():
                    dtheta = v['angle'] - e['angle']
                    if (abs(dtheta) < 0.25*np.pi) | (abs(dtheta) > 1.75*np.pi):
                        v_out.append(v['v'])
                    if (abs(dtheta) > np.pi*0.75) & (abs(dtheta) < np.pi*1.25):
                        v_in.append(v['v'])
                
                if len(v_out) > 0:
                    edges.at[eidx, "ncar_out"] = len(v_out)
                    v_avg_out, v_std_out = np.mean(v_out), np.std(v_out)
                    edges.at[eidx, "v_avg_out"] = v_avg_out
                    edges.at[eidx, "v_std_out"] = v_std_out
                if len(v_in) > 0:
                    edges.at[eidx, "ncar_in"] = len(v_in)
                    v_avg_in, v_std_in = np.mean(v_in), np.std(v_in)
                    edges.at[eidx, "v_avg_in"] = v_avg_in
                    edges.at[eidx, "v_std_in"] = v_std_in
                    
        # Add to arrays
        isnap = (day*info["nTG"]) + tg
        node_feat_arr[isnap] = nodes[["ncar","v_avg","v_std"]].to_numpy()
        edge_feat_arr[isnap] = edges[["ncar_out","v_avg_out","v_std_out",
                                      "ncar_in","v_avg_in","v_std_in"]].to_numpy()
        glbl_arr[isnap] = np.array([day,tg])
        
    print("Done writing day",day)

print("Saving",node_fname)
np.save(node_fname, node_feat_arr)
print("Saving",edge_fname)
np.save(edge_fname, edge_feat_arr)
print("Saving",send_fname)
np.save(send_fname, send_arr)
print("Saving",receive_fname)
np.save(receive_fname, rece_arr)

# Clear memory
del node_feat_arr, edge_feat_arr, send_arr, rece_arr

# To reload use
# array = np.load(fname+".npy", mmap_mode='r')

