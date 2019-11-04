import numpy as np
import pandas as pd
import os
import sys, getopt
from datetime import date
import time

long2km = 1/0.011741652782473
lat2km = 1/0.008994627867046
twopi = np.pi*2.

def stdout(s):
    sys.stdout.write(str(s)+'\n')

def add(fname, df, tcutoff, minvel, silent=True):
    '''
    Function for dumping dataframe to file
    df is a pandas DataFrame and should be
    pd.DataFrame(data=rawdata, columns=['x','y','timegroup','day','timeU70','ID'])
    We assume df values for x,y are in the desired domain
    fname header: d tg x y vx vy v id
    '''
    def get_row(i):
        t, x, y, ID = df['timeU70'][i], df['x'][i], df['y'][i], int(df['ID'][i])
        day, tg = int(df['day'][i]), int(df['timegroup'][i])
        return [t,x,y,ID,day,tg]
        

    N = (long)(len(df.index))
    oldN = (long)(len(df.index))
    nmissed = 0L # filtered out paths
    m = N/20
    if m < 1: m = 1
    dx, dy, dt, vx, vy = [],[],[],[],[]

    fout = open(fname,'a')

    if not silent: print "Stage 1"
    # Collect first point
    t0, x0, y0, ID0, day0, tg0 = get_row(0)
    i=0
    while (i<N-2):
        i+=1
        if (i%m==0): 
            if not silent:
                sys.stdout.write(str(int(100*float(i)/N))+"%..")
                sys.stdout.flush()

        # Collect second point
        t1, x1, y1, ID1, day1, tg1 = get_row(i)
        
        # Filter paths with too large a dT
        # t1,t0 are ms, dT is in minutes
        dT = (t1 - t0) / 60000.
        if ((dT > tcutoff) or (ID0 != ID1)): 
            nmissed += 1
            t0, x0, y0, ID0, day0, tg0 = t1, x1, y1, ID1, day1, tg1
            continue

        dX = x1 - x0
        dY = y1 - y0
        vx = dX / (dT/60.) # km/hr
        vy = dY / (dT/60.)

        # Filter paths that are too slow
        v = np.sqrt(vx*vx + vy*vy)
        if v<minvel:
            nmissed += 1
            t0, x0, y0, ID0, day0, tg0 = t1, x1, y1, ID1, day1, tg1
            continue

        # Write to file
        fout.write("%d %d %g %g %g %g %g %d\n" %(day0,\
            tg0, float(x0), float(y0),
            vx, vy, v, ID0))

        # Second pt is new ref point
        t0, x0, y0, ID0, day0, tg0 = t1, x1, y1, ID1, day1, tg1

    fout.close()
    N = N-nmissed
    if not silent:
        print "\nAdded " + str(N) + " paths from " + str(oldN)

    return int(N)


global_start = time.time()

arglen = len(sys.argv[1:])
arglist = sys.argv[1:]

tcutoff = 1.0
velmin = 0.0
runname = ""
runpath = "/home/michael/msc/summer17/traffic/traffic-stgcnn/veldata/"
tglen = 10

try:
    opts, args = getopt.getopt(arglist,"t:v:l:",["tcutoff=","velmin=",\
        "runpath=","runname=","tglen="])
except:
    stdout("Error in opt retrival...")
    sys.exit(2)

for opt, arg in opts:
    if opt in ("--timecutoff","-t"):
        tcutoff = float(arg)
        print "tcutoff ",tcutoff
    elif opt in ("--velmin","-v"):
        velmin = float(arg)
        print "velmin ",velmin
    elif opt == "--runpath":
        runpath = arg
        print "runpath ",runpath
    elif opt == "--runname":
        runname = arg
        print "runname ",runname
    elif opt in ("--tglen", "-l"):
        tglen = int(arg)
        print "tglen ",tglen



# Fifth ring
xmin = 116.1904 * long2km
xmax = 116.583642 * long2km
ymin = 39.758029 * lat2km
ymax = 40.04453 * lat2km
dxCell, dyCell = 1.0, 1.0 #in km

# Second ring
xmin = 116.33085226800 * long2km
xmax = 116.44826879600 * long2km
ymin = 39.85573366870 * lat2km
ymax = 39.96366920310 * lat2km

# Fir testing
#xmin *= 1.4
#xmax *= 1.4
#ymin *= 1.4
#ymax *= 1.4


sourcename = "/home/michael/msc/summer17/traffic/data/OUT0_FiveRing150buffer"

nTG = 60*24/tglen

finfo = open(runpath+runname+".info",'w')
finfo.write("xmin "+str(xmin)+"\n")
finfo.write("xmax "+str(xmax)+"\n")
finfo.write("ymin "+str(ymin)+"\n")
finfo.write("ymax "+str(ymax)+"\n")
finfo.write("tcutoff "+str(tcutoff)+"\n")
finfo.write("velmin "+str(velmin)+"\n")
finfo.write("tglen "+str(tglen)+"\n")
finfo.write("nTG "+str(nTG)+"\n")
finfo.write("source "+sourcename+"\n")
finfo.close()

fsource = open(sourcename, 'r')
fout = open(runpath+runname,'w')
fout.close() # clear existing file
#fout = open(outpath,'a')
# nninput file header: d tg x y vx vy v id

buffersize = int(1e5)
rawdata = np.empty(shape=[buffersize,6])
cnt_i = 0
cnt_itot = 0
cnt_dr = 0
cnt_iter = 0
cnt_success = 0
maxdrivers = int(1e8) 
maxdata = int(1e5)
driverIDs = {}
totaldrivers = 189515 # total num lines in OUT0

t0 = time.time()
t1 = time.time()
stdout("Processing source file")


# For testing and seeing
# how far apart this data ranges in time
# 1445207734622 is ms from first entry
mintime, maxtime = 1445207734622,0


for d in fsource.readlines():
    if cnt_dr%5e3==0:
        t2 = time.time()
        stdout(str(cnt_dr)+" drivers scanned of "+str(totaldrivers)) 
        stdout(str(t2-t1)+" seconds for last 5000 drivers, "+str(t2-t0)+" total time")
        t1 = time.time()

    # Limit amnt of data to speed up testing
    #if (cnt_itot >= maxdata) or (cnt_dr >= maxdrivers): break

    driver = d.split("  ")
    driverIDs.update({cnt_dr: driver[0]})
    driverdata = driver[1].split("|")
    for pt in driverdata:
        spt = pt.split(",")
        spt[0] = float(spt[0])*long2km
        spt[1] = float(spt[1])*lat2km
        # reject those outside range
        if (spt[0] > xmax) or (spt[0] < xmin): continue
        if (spt[1] > ymax) or (spt[1] < ymin): continue
        cnt_i+=1
        cnt_itot+=1
        itime = str.rsplit(spt[3])[1]
        itime = str.rsplit(itime, ":")
        itime = [int(t) for t in itime]
        timegroup = int((itime[0]*60 + itime[1])/tglen)
            
        idate = str.rsplit(spt[3])[0]
        idate = str.rsplit(idate, "-")
        idate = [int(d) for d in idate]
        day = (date(idate[0], idate[1], idate[2]).isoweekday()) - 1 # monday = 0, sunday = 6

            
        #rawdata[cnt_i-1] = [float(spt[0]), float(spt[1]), timegroup, day, long(spt[2]), cnt_dr]    
        msU70 = long(spt[2])
        if msU70 < mintime: mintime=msU70
        if msU70 > maxtime: maxtime=msU70
        
        rawdata[cnt_i-1] = [cnt_dr, float(spt[0]), float(spt[1]), long(spt[2]), timegroup, day]    

        if (cnt_i)%buffersize==0:
            rawdata = rawdata[np.lexsort(rawdata.T)]
            df = pd.DataFrame(data=rawdata, columns=['ID','x','y','timeU70','timegroup','day'])
            t4 = time.time()
            stdout("Adding rawdata...")
            cnt_success += add(runpath+runname,df,tcutoff,velmin,silent=True)
            t5 =time.time()
            stdout(str(t5-t4)+" seconds")
            df = None
            rawdata = np.empty(shape=[buffersize,6])
            stdout(str(cnt_i)+" points added")
            cnt_i=0

    cnt_dr+=1
    
print("mintime and maxtime in U70:",str(mintime)," ",str(maxtime))

fsource.close()

if cnt_i != 0:
    stdout("Adding rawdata remainder")
    rawdata = rawdata[:cnt_i]
    rawdata = rawdata[np.lexsort(rawdata.T)]
    df = pd.DataFrame(data=rawdata, columns=['ID','x','y','timeU70','timegroup','day'])
    cnt_success += add(runpath+runname,df,tcutoff,velmin,silent=True)
    df = None   
    rawdata = None
stdout("Done")

stdout(str(cnt_dr)+" drivers scanned\n"+str(cnt_success)+" points successfully added to "+runpath+runname)
finfo = open(runpath+runname+".info",'a')
finfo.write(str(cnt_dr)+" drivers scanned\n"+str(cnt_success)+" points successfully added to "+runpath+runname+"\n")
finfo.close()

global_end = time.time()
stdout("Total time "+str(global_end-global_start)+" seconds")
