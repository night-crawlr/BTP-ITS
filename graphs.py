import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def get_date(s):
    # get date from Timestamp object
    # get day number from date
    return str(s.date()).split('-')[2]

def plot_week(station_path,d):
    file_path = station_path + d    
    X = []   # week
    Y1 = []  # truck flow
    Y2 = []  # Total - truck flow 

    # read data from excel file
    df = pd.read_excel(file_path)
    cols = df.columns

    # Apply map on df['Date'] to get date
    df['Day'] = df[cols[0]].map(get_date)


    df = df.groupby('Day').sum()
    #print(df)

    df['Total - Truck (Veh/5 Minutes)'] =  - df['Truck Flow (Veh/5 Minutes)'] + df['Flow (Veh/5 Minutes)']

    # plot both not Truck and Truck Flow in single bar graph with Day as x axis
    df.plot.bar(y=['Total - Truck (Veh/5 Minutes)', 'Truck Flow (Veh/5 Minutes)'], rot=0)
    plt.title(d)
    try:
        os.mkdir(station_path+'graphs')
    except FileExistsError:
        pass

    plt.savefig(station_path+'graphs/'+str(d.split('.')[0]) + '.png')
    
station_path = './Truck_Flow/402214/'
# list dirs in station_path
dirs = os.listdir(station_path)
for d in dirs:
    # check if d is file
    if os.path.isfile(station_path + d):
        plot_week(station_path ,d)
        print(d+' Done')
        # Create a gif using png files
        # convert -delay 100 -loop 0 *.png truck.gif
        