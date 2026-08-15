import numpy as np
import pandas as pd



def CountCrossLine(data):
    count_openhigh = 0
    count_openlow = 0
    count_closehigh = 0
    count_closelow = 0
    count_highlow = 0
    count_total = 0
    for item in data:
        tmplen = len(item)
        for i in range(tmplen):
            count_total = count_total + 1
            if item[i][0] > item[i][1]:
                count_openhigh = count_openhigh + 1
                #print("O>H: "+str(item[i][0])+" ; "+str(item[i][1]))
            if item[i][0] < item[i][2]:
                count_openlow = count_openlow + 1
                #print("O<L: " + str(item[i][0]) + " ; " + str(item[i][2]))
            if item[i][3] > item[i][1]:
                count_closehigh = count_closehigh + 1
            if item[i][3] < item[i][2]:
                count_closelow = count_closelow + 1
            if item[i][1] < item[i][2]:
                count_highlow = count_highlow + 1

    if count_total == 0:
        return 0,0,0,0,0
    else:
        #return count_openhigh/count_total,count_openlow/count_total,count_closehigh/count_total,count_closelow/count_total,count_highlow/count_total
        return count_openhigh , count_openlow , count_closehigh , count_closelow , count_highlow

