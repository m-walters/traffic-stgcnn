import numpy as np
import os
import pandas as pd

import logman
import fluxGrid

long2km = 1/0.011741652782473
lat2km = 1/0.008994627867046

if __name__ == "__main__":
    # Logger -- see the logman README for usage
    logfile = "run.log"
    sformat = '%(name)s : %(message)s'
    logger = logman.logman(__name__, "debug", sformat, logfile)
    logger.add_handler("info", "%(message)s")


    fullArea = True

    if fullArea:
        # From Liang's road spreadsheets
        # Approximately 150x150 km
        xmin = 115.5148074 * long2km
        xmax = 117.26431366500 * long2km
        ymin = 39.42848884480 * lat2km
        ymax = 40.67874211840 * lat2km
        dxCell, dyCell = 1., 1. #in km
    else: #just to fifth ring
        # From Liang's slides, the fifth ring
        # Approximately 30x30 km
        xmin = 116.1904 * long2km
        xmax = 116.583642 * long2km
        ymin = 39.758029 * lat2km
        ymax = 40.04453 * lat2km
        dxCell, dyCell = 0.1, 0.1 #in km

    #TEMPORARY
    dxCell = 0.5
    dyCell = 0.5
    fluxgrid = fluxGrid.fluxgrid([xmin,xmax,ymin,ymax],dxCell,dyCell,logfile)

    data_dir = "/home/michael/msc/summer17/traffic/sample_data/processed_samples/"
    all_dir = os.listdir(data_dir)
    Nf = len(all_dir)
    cnt = 0
    for a_file in all_dir:
        if cnt==10: break
        cnt += 1
        logger.printl("info","\nProcessing batch "+str(cnt)+ " of "+str(Nf)+", file "+a_file+"...")
        data = pd.read_csv(data_dir+a_file, skiprows=1, 
                           names=['long','lat','unix70ms','dt','timegroup','day'])
        N = (long)(len(data.index))
        data['long'] = data['long']*long2km
        data['lat'] = data['lat']*lat2km
        data = data.rename(columns={"long": "x", "lat": "y"}) #fluxgrid will want these
        fluxgrid.process_batch(data)
