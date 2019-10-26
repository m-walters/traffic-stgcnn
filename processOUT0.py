import numpy as np
from datetime import date

print "Note: This file has almost 200k drivers, and will produce that many files."
fin = open("data/OUT0_FiveRing150buffer", 'r')
outdir = "data/"
cnt_dr = 0
for f in fin.readlines():
    cnt_dr+=1
    if (cnt_dr > ): break
    driver = f.split("  ")
    driverid = driver[0]
    fout = open(outdir+driver[0],"w+")
    #print("processing driver "+str(driver[0])+"...")
    data = np.empty(shape=[0, 5])
    driverdata = driver[1].split("|")
    for line in driverdata:
        spt = line.split(",")
        itime = str.rsplit(spt[3])[1]
        itime = str.rsplit(itime, ":")
        itime = [int(t) for t in itime]
        timegroup = int((itime[0]*60 + itime[1])/30)
            
        idate = str.rsplit(spt[3])[0]
        idate = str.rsplit(idate, "-")
        idate = [int(d) for d in idate]
        day = date(idate[0], idate[1], idate[2]).isoweekday() # monday = 0, sunday = 6
            
        data = np.append(data, [[float(spt[0]), float(spt[1]), timegroup, day, long(spt[2])]], axis=0)
        
    data = data[np.lexsort(data.T)]
    fout.write("long, lat, timegroup_30m, day, unix70ms\n")
    for i in range(len(data)):
        fout.write('%f, %f, %d, %d, %d\n' %         (data[i][0],data[i][1],data[i][2], data[i][3], data[i][4]))
    fout.close()

print "Done"

