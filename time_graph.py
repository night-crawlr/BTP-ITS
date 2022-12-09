import os
from matplotlib import pyplot as plt
import pandas as pd

def plot_file(dir_path,s):
    file_path = dir_path + s
    data =pd.read_excel(file_path)

    cols = data.columns
    data['Day'] = data[cols[0]].apply(lambda x: x.day) 
    data['Time'] = data[cols[0]].apply(lambda x: x.hour + x.minute/60)
    data['Month'] = data[cols[0]].apply(lambda x: x.month)

    # Iterate over each day
    for day in data['Day'].unique():
        day_data = data[data['Day'] == day]
        # create a new figure for each day
        plt.figure()
        plt.plot(day_data['Time'], day_data[cols[-3]], label='Day {}'.format(day))
        plt.plot(day_data['Time'], day_data[cols[-4]], label='Day {}'.format(day))
        plt.xlabel('Time (hour)')
        plt.ylabel('Flow')
        plt.legend()
        # save as month and day
        plt.savefig(dir_path+'time_graphs/'+'month_{}_day_{}.png'.format(data['Month'].unique()[0], day))

dir_path = './Truck_Flow/414025/'
for s in os.listdir(dir_path):
    if s.endswith('.xlsx'):
        try:
            os.mkdir(dir_path+'time_graphs')
        except:
            pass
        plot_file(dir_path,s)