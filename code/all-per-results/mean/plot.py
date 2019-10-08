import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

def get_color(counter):
    if counter > 9:
        counter = 0
    else:
        counter += 1
    return 'C'+str(counter), counter

def smooth(x, N=30):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def save_file(data_mean, file_name):
    with open(file_name,'w') as f:
        for i in range(len(data_mean)):
            f.write("%d\n" % data_mean[i])

def plot_data(data_mean, data_std,title, color,label='',y_label=''):
    data_x = range(len(smooth(data_mean)))
    data_y_plus = smooth(data_mean+data_std)
    data_y_minus = smooth(data_mean-data_std)

    plt.plot(smooth(data_mean),color=color, label=label)
    plt.fill_between(data_x,data_y_plus,data_y_minus,color=color,alpha=0.2)
    plt.xlabel('Episodes',fontsize=20)
    plt.ylabel(y_label,fontsize=20)
    plt.title(title,fontsize=20)
    plt.grid(b=True, which='major', color='b', linestyle='-')
    plt.legend()

def read_files():
    j = 0
    fd_name = ARGS.file+"0.txt"
    print(fd_name)
    exists = os.path.isfile(fd_name)
    data_all = []#np.array([])
    while exists:
        f = open(fd_name,"r")
        data_tmp = f.readlines()[:-1]
        data     = []
        for i in range(len(data_tmp)):
            data.append(int(data_tmp[i]))
        j += 1
        fd_name = ARGS.file+"%d.txt" % j
        print(fd_name)
        exists = os.path.isfile(fd_name)
        # if not data_all.size:
        #     data_all = np.asarray(data)
        # else:
        #     data_all = np.vstack((data_all,np.asarray(data)))
        data_all.append(np.asarray(data))
    
    return np.asarray(data_all)

def read_file(file_name):
    f = open(file_name,"r")
    data_tmp = f.readlines()[:-1]
    data     = []
    for i in range(len(data_tmp)):
        data.append(int(data_tmp[i]))
    
    return np.asarray(data)

def plot_comb_experiments():
    counter = 0
    if not ARGS.common:
        print("in first")
        # file_name1 = ARGS.file + "_duration"
        episodes_data = read_files()
        print(episodes_data)
        data_mean1 = np.mean(episodes_data,axis=0)
        data_std1 = np.std(episodes_data,axis=0)

        save_file(data_mean1, 'mean_'+ARGS.newFile+'_durations.txt')
        save_file(data_std1, 'std_'+ ARGS.newFile+'_durations.txt')

        # file_name2 = ARGS.file + "_rewards"
        rewards_data = read_files()
        data_mean2 = np.mean(rewards_data,axis=0)
        data_std2 = np.std(rewards_data,axis=0)
        save_file(data_mean2, 'mean_'+ARGS.newFile+'_rewards.txt')
        save_file(data_std2, 'std_'+ARGS.newFile+'_rewards.txt')
    else:
        
        files = [f for f in os.listdir('.') if os.path.isfile(f)]
        print(files)
        for f in files:
            if '10000_' in f:
                if 'durations' in f:
                    if 'mean' in f:
                        print(f)
                        data_mean1 = read_file(f)
                        std_file = f.replace('mean','std')
                        std1 = read_file(std_file)
                        color, counter = get_color(counter)
                        #_PER_rank_MountainCar-v0_rewards
                        #label = f.replace('mean_','').replace('_durations.txt','').replace('_PER_rank_CartPole-v1','')
                        label = f.replace('10000_', '').replace('mean_', '').replace('_durations.txt', '').replace('_MountainCar-v0','')
                        if 'CombinedReplayMemory' in f or 'NaiveReplayMemory' in f:
                            label = label.replace('_prop','')
                        plot_data(data_mean1, std1, 'Episode durations per episode', color, label=label, y_label='Episodes duration')
        
        plt.show()
        plt.close()
        # TODO: Currently two legends appear in the second graph. Need to verify if introducing plt.close() solves this.
        counter = 0
        for f in files:
            if '10000_' in f:
                if 'rewards' in f:
                    if 'mean' in f:
                        print(f)
                        data_mean2 = read_file(f)
                        std_file = f.replace('mean','std')
                        std2 = read_file(std_file)
                        color, counter = get_color(counter)
                        label = f.replace('10000_', '').replace('mean_', '').replace('_rewards.txt', '').replace('_MountainCar-v0', '')
                        if 'CombinedReplayMemory' in f or 'NaiveReplayMemory' in f:
                            label = label.replace('_prop','')
                        #label = f.replace('mean_','').replace('_rewards.txt','').replace('_PER_rank_CartPole-v1','')
                        plot_data(data_mean2, std2, 'Rewards per episode', color, label=label, y_label='Rewards')
        plt.show()

def plot_buffer(buffer_sizes):
    j = 0
    fd_name = ARGS.file+"0.txt"
    exists = os.path.isfile(fd_name)
    data_all = []
    while exists:
        f = open(fd_name,"r")
        data_tmp = f.readlines()[:-1]
        data     = []
        for i in range(len(data_tmp)):
            data.append(int(data_tmp[i]))
        plt.plot(smooth(data), label = buffer_sizes[j])
        j += 1
        fd_name = ARGS.file+"%d.txt" % j
        exists = os.path.isfile(fd_name)
        data_all.append(np.asarray(data))

    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title(ARGS.newFile)
    plt.grid()
    plt.legend()
    plt.show()


def main():
    plot_comb_experiments()

    #buffer_sizes = [100,1000,10000,100000]
    #plot_buffer(buffer_sizes)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='_PER_rank_MountainCar-v0_durations', type=str,
                        help='name of file to read until numeric value') # 'NaiveReplayMemory_prop_durations', 
    parser.add_argument('--newFile', default='_PER_rank_MountainCar-v0',type=str, help='Name of the file')
    parser.add_argument('--common', action='store_false', help='generate common plot')
    ARGS = parser.parse_args()
    print(ARGS)
    main()