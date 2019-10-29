import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

long2km = 1/0.011741652782473
lat2km = 1/0.008994627867046
# Fifth ring
# xmin = 116.1904 * long2km
# xmax = 116.583642 * long2km
# ymin = 39.758029 * lat2km
# ymax = 40.04453 * lat2km

def dist_from(ref,rs):
    # ref is 1x2 ndarray
    # rs is list of vectors to compare, nx2
    return np.linalg.norm((ref-rs),axis=1)

def generate_edges(nodedict_, maxdist=1., maxnbr=8, np_nodecoords=None, maxdist_km=True):
    #
    # maxdist in km
    #
    if not np_nodecoords:
        np_nodecoords = np.array([node["coords"] for node in nodedict_.values()],dtype=np.float)
    
    # calculate distances
    coords_cp = np_nodecoords.copy()
    n_node = len(coords_cp)
    indexes = np.arange(n_node).reshape(n_node,1)
    coords_cp = np.append(coords_cp,indexes,axis=1)
    z = np.zeros((n_node,1)) 
    coords_cp = np.append(coords_cp,z,axis=1)
    
    # Convert angular coords to km
    coords_cp[:,0] *= long2km
    coords_cp[:,1] *= lat2km
        
    for i,node in enumerate(np_nodecoords):
        edgerefs = []
        cent = np.asarray([node[0]*long2km,node[1]*lat2km])
        dists = dist_from(cent,coords_cp[:,:2])
        coords_cp[:,-1] = dists
        coords_cp = coords_cp[coords_cp[:,-1].argsort()]
        
        # Get nbr idxs within maxdist, up to maxnbr
        for j in range(maxnbr):
            if coords_cp[j+1,-1] > maxdist: break
            else:
                edgerefs.append(int(coords_cp[j+1,2]))
        nodedict_[i]["edgerefs"] = edgerefs

        
class graphplot:
    '''
    At some point you should create a function to zoom in on certain regions
    
    Possibly a very useful example if wanting to implement a live inset viewer:
    https://matplotlib.org/3.1.1/users/event_handling.html
    '''
    def __init__(self,nodedict,idxlist=None,figsize=None,figure=None,axis=None):
        if len(idxlist) == 0:
            idxlist = np.arange(len(nodedict))
        self.idxlist = idxlist
        self.nodes = {}
        self.edges = {}
        for idx in idxlist:
            # Note that the edge method double counts edges
            self.nodes.update({idx:nodedict[idx]})
            eidxs = nodedict[idx]["edgerefs"]
            new_eidxs = []
            for ei in eidxs:
                if ei in idxlist: new_eidxs.append(ei)
            self.edges.update({idx:list(new_eidxs)})
        self.npnodes = np.array([node["coords"] for node in self.nodes.values()],dtype=np.float)
        self.nnodes = len(self.npnodes)
        self.xlims = [np.min(self.npnodes[:,0]),np.max(self.npnodes[:,0])]
        self.ylims = [np.min(self.npnodes[:,1]),np.max(self.npnodes[:,1])]
        self.graphwidth = self.xlims[1] - self.xlims[0]
        self.graphheight = self.ylims[1] - self.ylims[0]
        xmargin = self.graphwidth * 0.1
        ymargin = self.graphheight * 0.1
        self.xlims = [self.xlims[0]-xmargin,self.xlims[1]+xmargin]
        self.ylims = [self.ylims[0]-ymargin,self.ylims[1]+ymargin]
        self.graphwidth *= 1.2
        self.graphheight *= 1.2
        self.nodecolor = "darkblue"
        self.noderadius = self.graphwidth / self.nnodes
        
        if not axis or not figure:
            self.fig, self.ax = plt.subplots(1,1,figsize=(20,20) if not figsize else figsize)
        else:
            self.fig, self.ax = figure, axis
        self.ax.set_ylim(self.ylims[0], self.ylims[1])
        self.ax.set_xlim(self.xlims[0], self.xlims[1])
        self.ax.set_aspect('equal', adjustable='box',anchor="NW")
        
    def drawgraph(self):
        for i in self.idxlist:
            node, edges = self.nodes[i]["coords"], self.edges[i]
            for edge in edges:
                nbr = self.nodes[edge]["coords"]
                self.plotLine(self.ax,node[0],node[1],nbr[0],nbr[1])
            self.plotnode(self.ax,node[0], node[1])            
        
    def plotnode(self,ax,x,y):
        circ = Circle((x,y),self.noderadius,color="b",alpha=0.8)
        ax.add_patch(circ)        
        
    def plotLine(self,ax,x1,y1,x2,y2,c='slategrey',lw=1.0,alpha=1.0):
        ax.plot([x1, x2], [y1, y2], color=c, linestyle='-', linewidth=lw, alpha=alpha);
        
    def savefig(self,path):
        self.fig.savefig(path)
        
    def get_rect_coords(self,rect):
        # returns [xmin,ymin,xmax,ymax]
        return rect.get_bbox().get_points().flatten()

        
class viewer(graphplot):
    '''
    Do:
    viewer = viewer(window)
    viewer.connect()
    To instantiate and use
    '''
    def __init__(self, nodedict, window=None, figsize=None, idxlist=None):
        #
        # window are the [xmin,xmax,ymin,ymax] dimensions of the viewer
        #
        self.figsize = (20,20) if not figsize else figsize
        self.fig_main = plt.figure(figsize=self.figsize)
        self.grid = plt.GridSpec(2,2,hspace=0.1, wspace=0.1,width_ratios=[2.2,1],height_ratios=[0.8,1])
        self.axs = [self.fig_main.add_subplot(self.grid[:,0])]
        self.axs.append(self.fig_main.add_subplot(self.grid[0,1]))

        # Initialize parent graph object
        super().__init__(nodedict,idxlist,(self.figsize[0]*0.75,self.figsize[1]*0.75), self.fig_main, self.axs[0])
        self.axs[0].set_aspect('equal', adjustable='box',anchor="NW")

        # Draw main graph
        self.drawgraph()
        
        # Setup viewer
        if not window:
        #    window = [self.xlims[0]+0.4*self.graphwidth,
        #              self.xlims[0]+0.6*self.graphwidth,
        #              self.ylims[0]+0.4*self.graphheight,
        #              self.ylims[0]+0.6*self.graphheight]
            dw = 2. / long2km
            window = [self.xlims[0] + 0.5*self.graphwidth - dw,
                      self.xlims[0] + 0.5*self.graphwidth + dw,
                      self.ylims[0] + 0.5*self.graphheight - dw,
                      self.ylims[0] + 0.5*self.graphheight + dw]
        xmin, xmax, ymin, ymax = window[0], window[1], window[2], window[3]
        self.window = Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,fill=False,linewidth=1.6,edgecolor="orangered")
        self.axs[0].add_patch(self.window)
        
        # Create inset axis
        xmin,ymin,xmax,ymax = self.get_rect_coords(self.window)
        self.axs[1].set_ylim(ymin,ymax)
        self.axs[1].set_xlim(xmin,xmax)
        self.axs[1].set_aspect('equal', adjustable='box',anchor="NE")
        self.axs[1].set_xticks([])
        self.axs[1].set_yticks([])
        self.press = None
        
        self.updateinset()
        
    def updateinset(self):
        # Remove any artists if present
        self.axs[1].clear()
        
        # Collect which nodes are included
        insetnodes = {}
        xmin,ymin,xmax,ymax = self.get_rect_coords(self.window)        
        for key,node in self.nodes.items():
            x,y = node["coords"]
            if x>xmin and x<xmax and y>ymin and y<ymax:
                insetnodes.update({key: node})
                self.plotnode(self.axs[1],x,y)
                edges = node["edgerefs"]
                for e in edges:
                    if e in self.idxlist:
                        nbr = self.nodes[e]["coords"]
                        self.plotLine(self.axs[1],x,y,nbr[0],nbr[1])

        self.axs[1].set_ylim(ymin,ymax)
        self.axs[1].set_xlim(xmin,xmax)
        #self.axs[1].set_aspect('equal', adjustable='box',anchor="NE")
        self.axs[1].set_xticks([])
        self.axs[1].set_yticks([])

#         pickle.dump(self., file('myplot.pickle', 'w'))        
        
    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.window.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.window.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.window.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        
    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.window.axes: return

        contains, attrd = self.window.contains(event)
        self.window.set_x(event.xdata - self.window.get_width()*0.5)
        self.window.set_y(event.ydata - self.window.get_height()*0.5)
        if True:
            # Update attributes for motion
            print('event contains', self.window.xy)
            x0, y0 = self.window.xy
            self.press = x0, y0, event.xdata, event.ydata

    def on_motion(self, event):
        'on motion we will move the window if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.window.axes: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        #print('x0=%f, xpress=%f, event.xdata=%f, dx=%f, x0+dx=%f' %
        #      (x0, xpress, event.xdata, dx, x0+dx))

        self.window.set_x(x0+dx)
        self.window.set_y(y0+dy)
        self.window.figure.canvas.draw()


    def on_release(self, event):
        'on release we reset the press data'
        self.press = None
        self.updateinset()
        self.window.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.window.figure.canvas.mpl_disconnect(self.cidpress)
        self.window.figure.canvas.mpl_disconnect(self.cidrelease)
        self.window.figure.canvas.mpl_disconnect(self.cidmotion)
