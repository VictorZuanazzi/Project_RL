import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
# import docx


def get_color(counter):
    # if counter > 9:
    #     counter = 0
    # else:
    counter += 1
    return 'C'+str(counter), counter

def smooth(x, N=30):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def save_file(data_mean, file_name):
    with open(file_name,'w') as f:
        for i in range(len(data_mean)):
            f.write("%d\n" % data_mean[i])

def plot_data(env_type, data_mean, data_std,title, color,label='',y_label=''):
    data_x = range(len(smooth(data_mean)))
    data_y_plus = smooth(data_mean+data_std)
    data_y_minus = smooth(data_mean-data_std)

    plt.rcParams["figure.figsize"] = (50,20)
    plt.plot(smooth(data_mean),color=color, label=label)
    plt.fill_between(data_x,data_y_plus,data_y_minus,color=color,alpha=0.2)
    plt.xlabel('Episodes ['+ env_type + ']')
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    # from matplotlib.font_manager import FontProperties
    #
    # fontP = FontProperties()
    # fontP.set_size('small')
    # plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, prop=fontP)
    # plt.tight_layout(rect=[0, 0, 0.75, 1])

    plt.legend(shadow=True)
    # plt.tight_layout()
    # plt.subplots_adjust(right=0.75)


def read_files(path, plot_type):

    files = os.listdir(path)
    docx_flag = False
    if any(['docx' in x for x in files]):
        selected_files = [x for x in files if ('docx' in x) and (plot_type in x)]
        docx_flag = True
    elif any(['txt' in x for x in files]):
        selected_files = [x for x in files if ('txt' in x) and (plot_type in x)]
    else:
        print("Bad file type")

    data_all = []
    for fd_name in selected_files:
        # if docx_flag == True:
        print(path + fd_name)
        f = open(path + fd_name, "r")
        data_tmp = f.readlines()[:-1]
        data = []
        for i in range(len(data_tmp)):
            data.append(int(data_tmp[i]))
        data_all.append(np.asarray(data))

    return np.asarray(data_all)

def read_file(file_name):
    f = open(file_name,"r")
    data_tmp = f.readlines()[:-1]
    data     = []
    for i in range(len(data_tmp)):
        data.append(int(data_tmp[i]))
    
    return np.asarray(data)

def plot_comb_experiments( env_type, env_path, parent_path):
    counter = 0

    # folders = ['CombinedReplayMemory-adapt', 'PER-rank',
    #            'NaiveReplayMemory', 'PER-prop',
    #            'PER-adapt-rank', 'NaiveReplayMemory-adapt', 'CombinedReplayMemory']
    folders = [f for f in os.listdir(env_path)]


    # files = [f for f in os.listdir('.') if os.path.isfile(f)]
    # file_name1 = ARGS.file + "_duration"

    if not ARGS.common:
        '''
        For Generating Mean and Standard deviation of multiple run
        '''

        for fol in folders:
            folder_path =  env_path + '/' + fol + '/results/'
            saving_path = env_path + '/' + fol + '/stats/'
            if not os.path.exists(saving_path):
                os.mkdir(saving_path)

            episodes_data = read_files(folder_path, "duration")

            print(env_path)
            if env_path == "../ER_results/LunarLander-v2":
                a = 1
            data_mean1 = np.mean(episodes_data,axis=0)
            data_std1 = np.std(episodes_data,axis=0)
        
            save_file(data_mean1, saving_path +'mean_'+ fol +'_durations.txt')
            save_file(data_std1, saving_path + 'std_'+ fol+'_durations.txt')

            # file_name2 = ARGS.file + "_rewards"
            rewards_data = read_files(folder_path, "rewards")
            data_mean2 = np.mean(rewards_data,axis=0)
            data_std2 = np.std(rewards_data,axis=0)
            save_file(data_mean2, saving_path +'mean_'+ fol +'_rewards.txt')
            save_file(data_std2, saving_path +'std_'+ fol +'_rewards.txt')
    else:
        plots_path = parent_path + '/' + 'plots'
        if not os.path.exists(plots_path+''):
            os.mkdir(plots_path)
        counter = 0
        # saving_path = env_path + '/'
        for fol in folders:
            stats_path = env_path + '/' + fol + '/stats/'

            files = os.listdir(stats_path)
            print(files)
            for f in files:
                f = stats_path + f
                if 'durations' in f:
                    if 'mean' in f:
                        data_mean1 = read_file(f)
                        std_file = f.replace('mean','std')
                        std1 = read_file(std_file)
                        color, counter = get_color(counter)
                        label = f.replace('mean_','').replace('_durations.txt','').replace(stats_path,'')
                        plot_data(env_type, data_mean1, std1, 'Episode durations per episode', color, label=label, y_label='Episodes duration')

        plt.savefig(plots_path + '/' + env_type + '_durations.png')
        plt.show()
        plt.close()

        counter = 0
        for fol in folders:
            stats_path = env_path + '/' + fol + '/stats/'
            files = os.listdir(stats_path)
            for f in files:
                f = stats_path + f
                if 'rewards' in f:
                    if 'mean' in f:
                        data_mean2 = read_file(f)
                        std_file = f.replace('mean','std')
                        std2 = read_file(std_file)
                        color, counter = get_color(counter)
                        label = f.replace('mean_','').replace('_rewards.txt','').replace(stats_path,'')
                        plot_data(env_type, data_mean2, std2, 'Rewards per episode', color, label=label, y_label='Rewards')
        plt.savefig(plots_path + '/' + env_type + '_rewards.png')
        plt.show()
        plt.close()

# def plot_buffer(buffer_sizes):
#     j = 0
#     fd_name = ARGS.plot_type+"0.txt"
#     exists = os.path.isfile(fd_name)
#     data_all = []
#     while exists:
#         f = open(fd_name,"r")
#         data_tmp = f.readlines()[:-1]
#         data     = []
#         for i in range(len(data_tmp)):
#             data.append(int(data_tmp[i]))
#         plt.plot(smooth(data), label = buffer_sizes[j])
#         j += 1
#         fd_name = ARGS.plot_type+"%d.txt" % j
#         exists = os.path.isfile(fd_name)
#         data_all.append(np.asarray(data))
#
#     plt.xlabel('Episodes')
#     plt.ylabel('Rewards')
#     plt.title(ARGS.newFile)
#     plt.grid()
#     plt.legend()
#     plt.show()


def main():

    parent_path = "../ER_results"




    # env_type = str(ARGS.env)

    envs = [
    'CartPole-v1',
    'MountainCar-v0',
    'LunarLander-v2'
    ]

    for env_type in envs:
        env_path = parent_path + "/" + env_type




        plot_comb_experiments(env_type, env_path, parent_path)

    # buffer_sizes = [100,1000,10000,100000]
    # plot_buffer(buffer_sizes)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # parser.add_argument('--plot_type', default='rewards', type=str,
    #                     help='name of file to read until numeric value') # 'rewards' or 'durations'

    # parser.add_argument('--newFile', default='CombinedReplay',type=str, help='Name of the file')

    parser.add_argument('--common', action='store_true', help='generate common plot. if use it the std/mean will be calculated'
                                                               ', otherwise they will be plotted')

    # ---------- path args ----------
    parser.add_argument('--replay', default='PER', type=str, choices=['CombinedReplayMemory',
                                                                      'NaiveReplayMemory', 'PER'],
                        help='type of experience replay')
    parser.add_argument('--env', default='CartPole-v1', type=str,
                        help='environments you want to evaluate')
    parser.add_argument('--pmethod', type=str, choices=['prop', 'rank'], default='prop',
                        help='proritized reply method: {prop or rank}')
    parser.add_argument('--buffer', default='10000', type=int,
                        help='buffer size for experience replay')

    ARGS = parser.parse_args()
    print(ARGS)

    # replay_path = env_path + "/" + str(ARGS.replay)
    # if ARGS.replay == 'PER':
    #     replay_path = replay_path + str(ARGS.pmethod)
    # images_path = replay_path + "/" + "images"
    # results_path = replay_path + "/" + "results"



    # print(files)
    main()