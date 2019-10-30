import numpy as np
import pandas as pd
import os
import sys, getopt
from datetime import date
import time
import inputadd

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
    N = (long)(len(df.index))
    oldN = (long)(len(df.index))
    nmissed = 0L # filtered out paths
    m = N/20
    dx, dy, dt, vx, vy = [],[],[],[],[]
    i = -1

    fout = open(fname,'a')
    if not silent: print "Stage 1"
    while (i<N-2):
        i+=1
        if (i%m==0): 
            if not silent:
                sys.stdout.write(str(int(100*float(i)/N))+"%..")
                sys.stdout.flush()
        dT = (df['timeU70'][i+1] - df['timeU70'][i]) / 60000.
        if ((dT > tcutoff) or (df['ID'][i] != df['ID'][i+1])): 
            nmissed += 1
            continue
        dX = df['x'][i+1] - df['x'][i]
        dY = df['y'][i+1] - df['y'][i]
        vx = dX / (dT/60.)
        vy = dY / (dT/60.)
        v = np.sqrt(vx*vx + vy*vy)
        if v<minvel:
            nmissed += 1
            continue

        day = int(df['day'][i])
        tg = int(df['timegroup'][i])
        fout.write("%d %d %g %g %g %g %g %d\n" %(int(df['day'][i]),\
            int(df['timegroup'][i]),
            float(df['x'][i]),
            float(df['y'][i]),
            vx,vy,v,int(df['ID'][i])))

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
runpath = "/home/michael/msc/summer17/traffic/data/"
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
maxdata = int(1e4)
driverIDs = {}
totaldrivers = 189515 # total num lines in OUT0

t0 = time.time()
t1 = time.time()
stdout("Processing source file")
for d in fsource.readlines():
    if cnt_dr%1e3==0:
        t2 = time.time()
        stdout(str(cnt_dr)+" drivers scanned of "+str(totaldrivers)) 
        stdout(str(t2-t1)+" seconds for last 1000 drivers, "+str(t2-t0)+" total time")
        t1 = time.time()
    #if (cnt_itot > maxdata) or (cnt_dr > maxdrivers): break
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

            
        #rawdata = np.append(rawdata, [[float(spt[0]), float(spt[1]), 
        #                               timegroup, day, long(spt[2]), cnt_dr]], axis=0)    
        rawdata[cnt_i-1] = [float(spt[0]), float(spt[1]), timegroup, day, long(spt[2]), cnt_dr]    

        if (cnt_i)%buffersize==0:
            rawdata = rawdata[np.lexsort(rawdata.T)]
            df = pd.DataFrame(data=rawdata, columns=['x','y','timegroup','day','timeU70','ID'])
            t4 = time.time()
            stdout("Dumping rawdata to nninput...")
            cnt_success += inputadd.add(runpath+runname,df,tcutoff,velmin,silent=True)
            t5 =time.time()
            stdout(str(t5-t4)+" seconds")
            df = None
            rawdata = np.empty(shape=[buffersize,6])
            stdout(str(cnt_i)+" points added")
            cnt_i=0

    cnt_dr+=1
    

fsource.close()
stdout("Adding rawdata remainder")
rawdata = rawdata[:cnt_i]
rawdata = rawdata[np.lexsort(rawdata.T)]
df = pd.DataFrame(data=rawdata, columns=['x','y','timegroup','day','timeU70','ID'])
cnt_success += inputadd.add(runpath+runname,df,tcutoff,velmin,silent=True)
df = None   
rawdata = None
stdout("Done")

stdout(str(cnt_dr)+" drivers scanned\n"+str(cnt_success)+" points successfully added to "+runpath+runname)
finfo = open(runpath+runname+".info",'a')
finfo.write(str(cnt_dr)+" drivers scanned\n"+str(cnt_success)+" points successfully added to "+runpath+runname+"\n")
finfo.close()

global_end = time.time()
stdout("Total time "+str(global_end-global_start)+" seconds")
