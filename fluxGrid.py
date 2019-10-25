import numpy as np
import pandas as pd

import logman

def linear(x,a,b):
    return a*x + b

def invert(a,b):
    return 1/float(a), -b/(float)(a)

class fluxgrid:
    '''
    Creates flux cell grid. Takes ranges=[ranges[0],ranges[1],ranges[2],ranges[3]]
    and cell widths as arguments, both in km.
    '''
    def __init__(self, ranges, dxCell=None, dyCell=None, logfile="run.log"):
        # Logger 
        sformat = '%(name)s : %(message)s'
        self.logger = logman.logman(__name__, "debug", sformat, logfile)
        self.logger.add_handler("info", "%(message)s")
        
        self.xpos = np.arange(ranges[0], ranges[1], dxCell) + 0.5*dxCell
        self.ypos = np.arange(ranges[2], ranges[3], dyCell) + 0.5*dyCell
        self.Nx, self.Ny = len(self.xpos), len(self.ypos)
        self.dxCell, self.dyCell = dxCell, dyCell

        # 6-tuple of (xpos, ypos, N, E, S, W), NESW fluxes
        self.cells = np.zeros((self.Nx, self.Ny, 6))
        for x in range(self.Nx):
            for y in range(self.Ny):
                self.cells[x,y,0], self.cells[x,y,1] = self.xpos[x], self.ypos[y]
                    
    def process_batch(self, data):
        '''
        data is a pandas DataFrame with at least columns 'x' and 'y'
        of the driver points in km
        '''
        N = (long)(len(data.index))
        a = [] #slopes
        b = [] #intercepts
        for i in range(0,N-1):
            # Notice, some numbers are NANs because of unchanging coordinates
            if (data['x'][i+1]-data['x'][i]) != 0.: # don't divide by 0
                a.append((data['y'][i+1]-data['y'][i])/(data['x'][i+1]-data['x'][i]))
                b.append(data['y'][i] - a[i]*data['x'][i])
            else:
                a.append(float('nan'))
                b.append(float('nan'))
                continue
        # Last data point has no line as it goes nowhere
        a.append(float('nan'))
        b.append(float('nan'))
        data['a'] = a
        data['b'] = b

        dx, dy = [],[]
        for i in range(0,N-1):
            dx.append(data['x'][i+1] - data['x'][i])
            dy.append(data['y'][i+1] - data['y'][i])
        dx.append(0.)
        dy.append(0.)
        data['dx'] = dx
        data['dy'] = dy
        
        nmissed = 0L # paths unable to process
        nshort = 0L # paths that did not leave a cell
        for i in range(0,N-1):
            try:
                xx, yy = 0, 0
                xxs = []
                short = True

                while(self.xpos[xx] + 0.5*self.dxCell < data['x'][i]): xx+=1
                # Starting point, cells with xpos[xx]
                xxs.append(xx)

                # Count how many edges it crosses
                if data['dx'][i] > 0: #traveling east
                    while(self.xpos[xx] + 0.5*self.dxCell < data['x'][i+1]):
                        xx += 1
                        xxs.append(xx)
                        short = False
                if data['dx'][i] < 0: #traveling west
                    while((self.xpos[xx]) - 0.5*self.dxCell > data['x'][i+1]):
                        xx -= 1
                        xxs.append(xx)
                        short = False

                # Locate starting yy index
                while(self.ypos[yy] + 0.5*self.dyCell < data['y'][i]): yy+=1

                # Need to find edges for each xx column
                for xx in xxs:
                    # Find the maximum y value of the line in the given xx column
                    if data['dx'][i] > 0:
                        y = linear(self.xpos[xx] + 0.5*self.dxCell, data['a'][i], data['b'][i])
                    if data['dx'][i] < 0:
                        y = linear(self.xpos[xx] - 0.5*self.dxCell, data['a'][i], data['b'][i])

                    # Don't want to go past the finish point
                    if xx == xxs[-1]: y = data['y'][i+1]

                    # Count number of edges between starting yy and y
                    if data['dy'][i] > 0:
                        while(self.ypos[yy] + 0.5*self.dyCell < y):
                            self.cells[xx, yy, 2] += 1
                            yy += 1
                            short = False

                    if data['dy'][i] < 0:
                        while(self.ypos[yy] - 0.5*self.dyCell > y):
                            self.cells[xx, yy, 4] += 1
                            yy -= 1
                            short = False

                    if xx != xxs[-1]:
                        if data['dx'][i] > 0:
                            self.cells[xx, yy, 3] += 1
                        else:
                            self.cells[xx, yy, 5] += 1

                if short: nshort+=1

            except:
                nmissed+=1

        self.logger.printl("info", "Successfully processed " + str(N-1-nmissed) + 
                      "/" + str(N-1) + " paths")
        self.logger.printl("info", "Of those processed, " + str(nshort) + 
                      " paths were too short (did not leave a cell)")

