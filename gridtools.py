from matplotlib.patches import Circle, Wedge, Polygon, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import numpy as np

twopi = np.pi*2.
long2km = 1/0.011741652782473
lat2km = 1/0.008994627867046
dxCell, dyCell = 1.0,1.0 #in km

class gridtools:
    def __init__(self,ranges,Nx,Ny,nTG,tglen):
        self.Nx = Nx
        self.Ny = Ny
        self.nTG = nTG
        self.tglen = tglen
        xmin = ranges[0]
        ymin = ranges[1]
        xmax = ranges[2]
        ymax = ranges[3]

        # Since the main run files are fifth ring
        # Fifth ring
        xmin5 = 116.1904 * long2km
        xmax5 = 116.583642 * long2km
        ymin5 = 39.758029 * lat2km
        ymax5 = 40.04453 * lat2km
        
        self.Nxmin, self.Nxmax = int((xmin-xmin5)//dxCell), int((xmax-xmin5)//dxCell)
        self.Nymin, self.Nymax = int((ymin-ymin5)//dyCell), int((ymax-ymin5)//dyCell)

    def make_colormap(self,seq):
        """Return a LinearSegmentedColormap
        seq: a sequence of floats and RGB-tuples. The floats should be increasing
        and in the interval (0,1).
        """
        seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
        cdict = {'red': [], 'green': [], 'blue': []}
        for i, item in enumerate(seq):
            if isinstance(item, float):
                r1, g1, b1 = seq[i - 1]
                r2, g2, b2 = seq[i + 1]
                cdict['red'].append([item, r1, r2])
                cdict['green'].append([item, g1, g2])
                cdict['blue'].append([item, b1, b2])
        return mcolors.LinearSegmentedColormap('CustomMap', cdict)



    def read_states(self,fname,D,TG,order=None):
        # states[x][y][d][tg][th] -- old
        # states[x][y][th] -- new
        #
        # Need to modify to read only specific day and tg
        #
        # source file should be day,tg,x,y,th
        #
        states = [[[[] for th in range(8)] for y in range(self.Ny)] \
                            for x in range(self.Nx)]
        fin = open(fname,'r')
        # Use end of first line to count states
        nstate = 0
        spt = fin.readline().split()
        for st in spt[5:]:
            nstate+=1
        fin.seek(0) # return back to top
        npertg = 0
        for l in fin.readlines():
            spt = l.split()
            tg = int(spt[1])
            if tg < 1:
                npertg+=1
            else: break
        fin.seek(0)

        # Skip along until proper day and tg
        nskip = D * (self.nTG*npertg) + TG*npertg
        iskip = 0

        for l in fin.readlines():
            if iskip < nskip:
                iskip+=1
                continue
            spt = l.split()
            d,tg,x,y,th = int(spt[0]),int(spt[1]),int(spt[2]),int(spt[3]),int(spt[4])
            if (x in range(self.Nxmin,self.Nxmax)) and (y in range(self.Nymin,self.Nymax)):
                state = []
                for s in spt[5:]:
                    state.append(float(s))
                if order:
                    state = [state[i] for i in order]
                try:
                    states[x-self.Nxmin][y-self.Nymin][th]=state
                except:
                    print x,y,d,tg,th,state,self.Nxmin,self.Nymin

            # Don't need to keep going
            if (x>self.Nxmax) and (y>self.Nymax):
                break
        fin.close()


        # Sweep over states and make them [-1,-1] if empty
    #     for x in range(self.Nx):
    #         for y in range(self.Ny):
    #             for d in range(7):
    #                 for tg in range(self.nTG):
    #                     for th in range(8):
    #                         if not states[x][y][d][tg][th]:
    #                             s = [0 for _ in range(nstate)]
    #                             states[x][y][d][tg][th] = s

        return states



    def plotmap(self,axis,states,xcntrs,ycntrs,d,tg):
        w = xcntrs[1] - xcntrs[0]
        for xi in range(len(xcntrs)):
            for yi in range(len(ycntrs)):
                x,y = xcntrs[xi],ycntrs[yi]
                self.plotcell(axis,[x,y],w,states[xi][yi])


    def diffmap(self,axis,states_old,states_new,xcntrs,ycntrs,d,tg,ui,tag_old,tag_new,cols):
        # states are ndarrays that used to be nested lists
        # d and tg are lists [d_old,d_new],[tg_old,tg_new]
        # ui is which state to be shown
        a1, a2 = states_old, states_new
        statediff = (a2 - a1)

        w = xcntrs[1] - xcntrs[0]
        for xi in range(len(xcntrs)):
            for yi in range(len(ycntrs)):
                x,y = xcntrs[xi],ycntrs[yi]
                self.plotdiffcell(axis,[x,y],w,statediff[xi,yi,d[1],tg[1],:,ui],cols)


        days = {0:"Mon",
                1:"Tue",
                2:"Wed",
                3:"Thu",
                4:"Fri",
                5:"Sat",
                6:"Sun"}
        tgperhr = 60//self.tglen
        hr1 = str(tg[0]//tgperhr)
        hr2 = str(tg[1]//tgperhr)
        m1 = str((tg[0]%tgperhr)*self.tglen)
        m2 = str((tg[1]%tgperhr)*self.tglen)
        if m1=="0": m1="00"
        if m2=="0": m2="00"
        title = "state "+str(ui)+" | "+tag_old+" "+days[d[0]]+" "+hr1+":"+m1+\
            " -> "+tag_new+" "+days[d[1]]+" "+hr2+":"+m2 
        fsize = 16+(len(xcntrs)//2)
        axis.set_title(title,fontsize=fsize)



    def plotcell(self,axis,cntr,w,u):
        # w is full width of cell
        # u is list of c values for each slice
        # cntr is center of cell
        cx, cy = cntr[0],cntr[1]
        w2 = w*0.5
        rw = w*0.08 # roadwidth
        if len(u[0]) == 2:
            colors = [(0.1 + 0.9*c[1],0.1+0.6*c[0],0.1+0.4*c[0]) for c in u]
            colors = [(c[1],c[0],0) for c in u]
            
#        if len(u[0]) == 3:
#            colors = [(c[1],0,c[0]) for c in u]

        ecol = 'k'

        # First color the background
        back = Rectangle((cx-w2,cy-w2),w,w,facecolor='k',edgecolor=ecol)
        axis.add_patch(back)

        thetas = [twopi * i / 8. for i in range(8)]


        l2 = rw*np.sin(twopi/16.)
        l3 = rw/np.tan(twopi/16.)
        rightroads = [
            [[cx,cy],[cx+l3,cy-rw],[cx+w2,cy-rw],[cx+w2,cy]],
            [[cx,cy],[cx+rw,cy+l3],[cx+rw,cy+w2],[cx,cy+w2]],
            [[cx,cy],[cx-l3,cy+rw],[cx-w2,cy+rw],[cx-w2,cy]],
            [[cx,cy],[cx-rw,cy-l3],[cx-rw,cy-w2],[cx,cy-w2]],
        ]
        rightroads2 = [
            [[cx,cy],[cx-l3,cy-rw],[cx-w2,cy-rw],[cx-w2,cy]],
            [[cx,cy],[cx+rw,cy-l3],[cx+rw,cy-w2],[cx,cy-w2]],
            [[cx,cy],[cx+l3,cy+rw],[cx+w2,cy+rw],[cx+w2,cy]],
            [[cx,cy],[cx-rw,cy+l3],[cx-rw,cy+w2],[cx,cy+w2]],
        ]

        l1 = np.sqrt(2)*rw
        diagroads = [
            [[cx,cy],[cx+l3,cy+rw],[cx+w2,cy+w2-l1],[cx+w2,cy+w2]],
            [[cx,cy],[cx-rw,cy+l3],[cx-w2+l1,cy+w2],[cx-w2,cy+w2]],
            [[cx,cy],[cx-l3,cy-rw],[cx-w2,cy-w2+l1],[cx-w2,cy-w2]],
            [[cx,cy],[cx+rw,cy-l3],[cx+w2-l1,cy-w2],[cx+w2,cy-w2]],
        ]
        # 0 is refl of 1 about x
        # 1 is refl of 2 about y
        # 2 is refl of 3 about x
        # 3 is refl of 0 about y
        diagroads2 = [
            [[cx,cy],[cx-rw,cy-l3],[cx-w2+l1,cy-w2],[cx-w2,cy-w2]],
            [[cx,cy],[cx+l3,cy-rw],[cx+w2,cy-w2+l1],[cx+w2,cy-w2]],
            [[cx,cy],[cx+rw,cy+l3],[cx+w2-l1,cy+w2],[cx+w2,cy+w2]],
            [[cx,cy],[cx-l3,cy+rw],[cx-w2,cy+w2-l1],[cx-w2,cy+w2]],
        ]

        for i in range(4):
            # corner wedges
            ii = i*2
            poly = Polygon(np.asarray(diagroads[i]), True, facecolor=colors[ii+1],edgecolor=ecol)
            axis.add_patch(poly)
            poly = Polygon(np.asarray(diagroads2[i]), True, facecolor=colors[ii+1],edgecolor=ecol)
            axis.add_patch(poly)
        for i in [0,2,4,6]:
            # flat wedges
            ii = i/2
            poly = Polygon(np.asarray(rightroads[ii]), True, facecolor=colors[i],edgecolor=ecol)
            axis.add_patch(poly)
            poly = Polygon(np.asarray(rightroads2[ii]), True, facecolor=colors[i],edgecolor=ecol)
            axis.add_patch(poly)



    def plotdiffcell(self,axis,cntr,w,u,cols):
        # w is full width of cell
        # u is c value diff for each slice
        # cntr is center of cell
        cx, cy = cntr[0],cntr[1]
        w2 = w*0.5
        rw = w*0.08 # roadwidth
        colors = []
        for c in u:
            if c>0:
                col = tuple([c1*c for c1 in cols[1]])
                colors.append(col)
            else:
                col = tuple([-c*c0 for c0 in cols[0]])
                colors.append(col)

        ecol = 'k'

        # First color the background
        back = Rectangle((cx-w2,cy-w2),w,w,facecolor='w',edgecolor=ecol)
    #     back = Rectangle((cx-w2,cy-w2),w,w,facecolor=(0,0.6,0.4),edgecolor=ecol)
        axis.add_patch(back)

        thetas = [twopi * i / 8. for i in range(8)]


        l2 = rw*np.sin(twopi/16.)
        l3 = rw/np.tan(twopi/16.)
        rightroads = [
            [[cx,cy],[cx+l3,cy-rw],[cx+w2,cy-rw],[cx+w2,cy]],
            [[cx,cy],[cx+rw,cy+l3],[cx+rw,cy+w2],[cx,cy+w2]],
            [[cx,cy],[cx-l3,cy+rw],[cx-w2,cy+rw],[cx-w2,cy]],
            [[cx,cy],[cx-rw,cy-l3],[cx-rw,cy-w2],[cx,cy-w2]],
        ]
        rightroads2 = [
            [[cx,cy],[cx-l3,cy-rw],[cx-w2,cy-rw],[cx-w2,cy]],
            [[cx,cy],[cx+rw,cy-l3],[cx+rw,cy-w2],[cx,cy-w2]],
            [[cx,cy],[cx+l3,cy+rw],[cx+w2,cy+rw],[cx+w2,cy]],
            [[cx,cy],[cx-rw,cy+l3],[cx-rw,cy+w2],[cx,cy+w2]],
        ]

        l1 = np.sqrt(2)*rw
        diagroads = [
            [[cx,cy],[cx+l3,cy+rw],[cx+w2,cy+w2-l1],[cx+w2,cy+w2]],
            [[cx,cy],[cx-rw,cy+l3],[cx-w2+l1,cy+w2],[cx-w2,cy+w2]],
            [[cx,cy],[cx-l3,cy-rw],[cx-w2,cy-w2+l1],[cx-w2,cy-w2]],
            [[cx,cy],[cx+rw,cy-l3],[cx+w2-l1,cy-w2],[cx+w2,cy-w2]],
        ]
        # 0 is refl of 1 about x
        # 1 is refl of 2 about y
        # 2 is refl of 3 about x
        # 3 is refl of 0 about y
        diagroads2 = [
            [[cx,cy],[cx-rw,cy-l3],[cx-w2+l1,cy-w2],[cx-w2,cy-w2]],
            [[cx,cy],[cx+l3,cy-rw],[cx+w2,cy-w2+l1],[cx+w2,cy-w2]],
            [[cx,cy],[cx+rw,cy+l3],[cx+w2-l1,cy+w2],[cx+w2,cy+w2]],
            [[cx,cy],[cx-l3,cy+rw],[cx-w2,cy+w2-l1],[cx-w2,cy+w2]],
        ]

        for i in range(4):
            # corner wedges
            ii = i*2
            poly = Polygon(np.asarray(diagroads[i]), True, facecolor=colors[ii+1],edgecolor=ecol)
            axis.add_patch(poly)
            poly = Polygon(np.asarray(diagroads2[i]), True, facecolor=colors[ii+1],edgecolor=ecol)
            axis.add_patch(poly)
        for i in [0,2,4,6]:
            # flat wedges
            ii = i/2
            poly = Polygon(np.asarray(rightroads[ii]), True, facecolor=colors[i],edgecolor=ecol)
            axis.add_patch(poly)
            poly = Polygon(np.asarray(rightroads2[ii]), True, facecolor=colors[i],edgecolor=ecol)
            axis.add_patch(poly)
