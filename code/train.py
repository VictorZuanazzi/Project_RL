import numpy as np
from environment import get_env
from model import *
from replay import *
import argparse
import torch
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm as _tqdm
from torch.autograd import Variable
import random, os

# ----device-----------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)


# using exponential decay rather than linear decay
# def get_epsilon(it):
#     # YOUR CODE HERE
#     return max(0.01,(-0.95/ARGS.decay_steps)*it + 1)

def get_beta(it, total_it, beta0):
    return beta0 + (it / total_it) * (1 - beta0)


def select_action(model, state, epsilon):
    state = torch.from_numpy(state).float()
    with torch.no_grad():
        actions = model(state.to(device))

    rand_num = np.random.uniform(0, 1, 1)
    if epsilon > rand_num:
        index = torch.randint(0, len(actions), (1, 1))
    else:
        value, index = actions.max(0)

    return int(index.item())


def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter 
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def compute_q_val(model, state, action):
    actions = model(state)
    return actions.gather(1, action.unsqueeze(1))


def compute_target(model_target, reward, next_state, done, discount_factor):
    # done is a boolean (vector) that indicates if next_state is terminal (episode is done)
    non_terminal_states_mask = torch.tensor([1 if not s else 0 for s in done])
    right_index = non_terminal_states_mask.nonzero().squeeze(1) if len(non_terminal_states_mask.nonzero().size()) > 1 \
        else non_terminal_states_mask.nonzero().squeeze(0)
    non_terminal_states = next_state[right_index]

    next_state_values = torch.zeros(done.size()[0]).to(device)
    if not non_terminal_states.nelement() == 0:
        next_state_values[right_index], _ = model_target(non_terminal_states).max(1)

    target = reward + discount_factor * next_state_values

    return target.detach().unsqueeze(1)

def train(model, model_target, memory, optimizer, batch_size, discount_factor, TAU, iter, beta=None):

    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # transition batch is taken from experience replay memory
    if ARGS.replay == 'PER':
        transitions, batch_idx, priorities = memory.sample(batch_size)
    else:
        transitions = memory.sample(batch_size)

    if type(transitions[0]) == int:
        return None
    # print(batch_idx)
    # transition is a list of 5-tuples, instead we want 5 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float).to(device)
    action = torch.tensor(action, dtype=torch.int64).to(device)  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float).to(device)
    reward = torch.tensor(reward, dtype=torch.float).to(device)
    done = torch.tensor(done, dtype=torch.uint8).to(device)  # Boolean

    # compute the q value
    q_val = compute_q_val(model, state, action)

    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_target(model_target, reward, next_state, done, discount_factor)

    if ARGS.replay == 'PER':
        w = (1 / (batch_size * np.array(priorities)) ** beta)
        w = torch.tensor(w, dtype=torch.float, requires_grad=False).to(device)

        if ARGS.norm:
            w = w / torch.max(w)

        loss = torch.mean(w * abs(q_val - target))
        td_error = target - q_val
        for i in range(batch_size):
            val = abs(td_error[i].data[0])
            memory.update(batch_idx[i], val)
    else:
        # loss is measured from error between current and newly expected Q values
        # loss = F.smooth_l1_loss(q_val, target)
        loss = F.mse_loss(q_val, target)

    # backpropagation of loss to Neural Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if ARGS.update_freq % iter == 0:
        soft_update(model, model_target, TAU)

    return loss.item()


def smooth(x, N=10):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def main():
    # update this disctionary as per the implementation of methods
    memory = {'NaiveReplayMemory': NaiveReplayMemory,
              'CombinedReplayMemory': CombinedReplayMemory,
              'PER': PrioritizedReplayMemory,
              'MinMax': MinMaxNaiveReplayMemory}


    if ARGS.adaptive_buffer:
        # Introduces the buffer manager for the adaptive buffer size.
        manage_memory = BufferSizeManager(initial_capacity=ARGS.buffer,
                                      size_change=ARGS.buffer_step_size)

    # environment
    env, (input_size, output_size) = get_env(ARGS.env)
    # env.seed(seed_value)

    network = {'CartPole-v1': CartNetwork(input_size, output_size, ARGS.num_hidden).to(device),
               'MountainCar-v0': MountainNetwork(input_size, output_size, ARGS.num_hidden).to(device),
               'LunarLander-v2': LanderNetwork(input_size, output_size, ARGS.num_hidden).to(device)}

    # create new file to store durations
    i = 0
    fd_name = "results/" + str(ARGS.buffer) + "_" + str(ARGS.replay) + "_" + str(
        ARGS.pmethod) + '_' + ARGS.env + "_durations0.txt"
    exists = os.path.isfile(fd_name)
    while exists:
        i += 1
        fd_name = "results/" + str(ARGS.buffer) + "_" + str(ARGS.replay) + "_" + str(
            ARGS.pmethod) + '_' + ARGS.env + "_durations%d.txt" % i
        exists = os.path.isfile(fd_name)
    fd = open(fd_name, "w+")

    # create new file to store rewards
    i = 0
    fr_name = "results/" + str(ARGS.buffer) + "_" + str(ARGS.replay) + "_" + str(
        ARGS.pmethod) + '_' + ARGS.env + "_rewards0.txt"
    exists = os.path.isfile(fr_name)
    while exists:
        i += 1
        fr_name = "results/" + str(ARGS.buffer) + "_" + str(ARGS.replay) + "_" + str(
            ARGS.pmethod) + '_' + ARGS.env + "_rewards%d.txt" % i
        exists = os.path.isfile(fr_name)
    fr = open(fr_name, "w+")

    # Save experiment hyperparams
    i = 0
    exists = os.path.isfile("results/" + str(ARGS.buffer) + "_" + str(ARGS.replay) + "_" + str(
        ARGS.pmethod) + '_' + ARGS.env + "_info0.txt")
    while exists:
        i += 1
        exists = os.path.isfile("results/" + str(ARGS.buffer) + "_" + str(ARGS.replay) + "_" + str(
            ARGS.pmethod) + '_' + ARGS.env + "_info%d.txt" % i)
    fi = open("results/" + str(ARGS.buffer) + "_" + str(ARGS.replay) + "_" + str(
        ARGS.pmethod) + '_' + ARGS.env + "_info%d.txt" % i, "w+")
    file_counter = i
    fi.write(str(ARGS))
    fi.close()

    # -----------initialization---------------
    if ARGS.replay == 'PER':
        replay = memory[ARGS.replay](ARGS.buffer, ARGS.pmethod)
        filename = "results/" + str(ARGS.buffer) + "_" + 'weights_' + str(
            ARGS.replay) + '_' + ARGS.pmethod + '_' + ARGS.env + "_%d.pt" % file_counter  # +'_.pt'
    else:
        replay = memory[ARGS.replay](ARGS.buffer)
        filename = "results/" + str(ARGS.buffer) + "_" + 'weights_' + str(
            ARGS.replay) + '_' + ARGS.env + "_%d.pt" % file_counter  # +'_.pt'

    model = network[ARGS.env]  # local network
    model_target = network[ARGS.env]  # target_network

    optimizer = optim.Adam(model.parameters(), ARGS.lr)

    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    rewards_per_episode = []

    scores_window = deque(maxlen=100)
    eps = ARGS.EPS
    # -------------------------------------------------------

    for i_episode in tqdm(range(ARGS.num_episodes), ncols=100):

        # Sample a transition
        s = env.reset()
        done = False
        epi_duration = 0
        r_sum = 0

        # for debugging purposes:
        if (ARGS.debug_mode):
            print(f"buffser size: {len(replay)}")

        render_env_bool = False
        if (ARGS.render_env > 0) and not (i_episode % ARGS.render_env):
            render_env_bool = True
            env.render()

        max_steps = 1000
        for t in range(max_steps):
            # eps = get_epsilon(global_steps) # Comment this to to not use linear decay

            model.eval()
            a = select_action(model, s, eps)

            model.train()
            s_next, r, done, _ = env.step(a)

            beta = None
            if (ARGS.replay == 'PER') or ARGS.adaptive_buffer:
                state = torch.tensor(s, dtype=torch.float).to(device).unsqueeze(0)
                action = torch.tensor(a, dtype=torch.int64).to(device).unsqueeze(0)  # Need 64 bit to use them as index
                next_state = torch.tensor(s_next, dtype=torch.float).to(device).unsqueeze(0)
                reward = torch.tensor(r, dtype=torch.float).to(device).unsqueeze(0)
                done_ = torch.tensor(done, dtype=torch.uint8).to(device).unsqueeze(0)
                with torch.no_grad():
                    q_val = compute_q_val(model, state, action)
                    target = compute_target(model_target, reward, next_state, done_, ARGS.discount_factor)
                td_error = F.smooth_l1_loss(q_val, target)

                # redefines memory size
                new_buffer_size = manage_memory.update_memory_size(td_error.item())
                replay.resize_memory(new_buffer_size)

            if (ARGS.replay == 'PER'):
                replay.push(td_error, (s, a, r, s_next, done))
                beta = get_beta(i_episode, ARGS.num_episodes, ARGS.beta0)
            else:
                replay.push((s, a, r, s_next, done))

            loss = train(model, model_target, replay, optimizer, ARGS.batch_size, ARGS.discount_factor, ARGS.TAU,
                         global_steps, beta=beta)

            s = s_next
            epi_duration += 1
            global_steps += 1

            if done:
                break

            r_sum += r
            # visualize
            if render_env_bool:
                env.render()

        eps = max(0.01, ARGS.eps_decay * eps)
        rewards_per_episode.append(r_sum)
        episode_durations.append(epi_duration)
        scores_window.append(r_sum)

        # store episode data in files
        fr.write("%d\n" % r_sum)
        fr.close()
        fr = open(fr_name, "a")

        fd.write("%d\n" % epi_duration)
        fd.close()
        fd = open(fd_name, "a")

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        # if np.mean(scores_window)>=200.0:
        #     print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
        # break

        # if epi_duration >= 500: # this value is environment dependent
        #     print("Failed to complete in trial {}".format(i_episode))

        # else:
        # print("Completed in {} trials".format(i_episode))
        # break

    # close files
    fd.close()
    fr.close()

    env.close()

    print(f"Saving weights to {filename}")
    torch.save({
        # You can add more here if you need, e.g. critic
        'policy': model.state_dict()  # Always save weights rather than objects
    },
        filename)

    plt.plot(smooth(episode_durations, 10))
    plt.title('Episode durations per episode')
    # plt.show()
    plt.savefig("images/" + str(ARGS.buffer) + "_" + str(
        ARGS.replay) + '_' + ARGS.pmethod + '_' + ARGS.env + '_Episode' + "%d.png" % file_counter)

    plt.plot(smooth(rewards_per_episode, 10))
    plt.title("Rewards per episode")
    # plt.show()
    plt.savefig("images/" + str(ARGS.buffer) + "_" + str(
        ARGS.replay) + '_' + ARGS.pmethod + '_' + ARGS.env + '_Rewards' + "%d.png" % file_counter)
    return episode_durations


def get_action(state, model):
    return model(state).multinomial(1)


def evaluate():
    if ARGS.replay == 'PER':
        filename = 'weights_' + str(ARGS.replay) + '_' + ARGS.pmethod + '_' + ARGS.env + '_.pt'
    else:
        filename = 'weights_' + str(ARGS.replay) + '_' + ARGS.env + '_.pt'

    env, (input_size, output_size) = get_env(ARGS.env)
    # set env seed
    env.seed(seed_value)

    network = {'CartPole-v1': CartNetwork(input_size, output_size, ARGS.num_hidden).to(device),
               'MountainCar-v0': MountainNetwork(input_size, output_size, ARGS.num_hidden).to(device),
               'LunarLander-v2': LanderNetwork(input_size, output_size, ARGS.num_hidden).to(device)}

    model = network[ARGS.env]
    model.eval()
    if os.path.isfile(filename):
        print(f"Loading weights from {filename}")
        # weights = torch.load(filename)
        weights = torch.load(filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(weights['policy'])
    else:
        print("Please train the model or provide the saved 'weights.pt' file")
    episode_durations = []
    for i in range(20):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            steps += 1
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float).to(device)
                action = get_action(state, model).item()
                state, reward, done, _ = env.step(action)
                env.render()
        episode_durations.append(steps)
        print(i)
    env.close()

    plt.plot(episode_durations)
    plt.title('Episode durations')
    plt.show()
    # plt.savefig('foo.png')


if __name__ == "__main__":

    path = "images"
    if not os.path.exists(path):
        os.mkdir(path)

    path2 = "results"
    if not os.path.exists(path2):
        os.mkdir(path2)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', default=1000, type=int,
                        help='max number of episodes')
    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--num_hidden', default=64, type=int,
                        help='dimensionality of hidden space')
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--discount_factor', default=0.8, type=float)
    parser.add_argument('--replay', default='PER', type=str, choices=['CombinedReplayMemory', \
                                                                      'NaiveReplayMemory', 'PER', 'MinMax'],
                        help='type of experience replay')
    parser.add_argument('--env', default='CartPole-v1', type=str,
                        help='environments you want to evaluate')
    parser.add_argument('--buffer', default='10000', type=int,
                        help='buffer size for experience replay')
    parser.add_argument('--buffer_type', default="static", type=str,
                        choices=['static', 'adaptive'],
                        help="chose from constant (static) and dynamic (adaptive) buffer size.")

    parser.add_argument('--beta0', default=0.4, type=float)
    parser.add_argument('--pmethod', type=str, choices=['prop', 'rank'], default='prop', \
                        help='proritized reply method: {prop or rank}')
    parser.add_argument('--TAU', default=1e-3, type=float, \
                        help='parameter for soft update of weight; set it to one for hard update')
    parser.add_argument('--EPS', default='1.0', type=float,
                        help='epsilon')
    parser.add_argument('--eps_decay', default=.995, type=float,
                        help='decay constant')
    parser.add_argument('--update_freq', default=500, help='Update frequence in steps of target network parametes')
    parser.add_argument('--norm', default='True', type=bool,
                        help="weight normalization: {True, False}")
    parser.add_argument('--render_env', default=0, type=int,
                        help='render environment once every number of steps, 0 does not render the environment')

    parser.add_argument('--seed_value', default=42, type=int,
                        help='seed to set in random, numpy and pytorch to ensure reproducibility')

    parser.add_argument('--debug_mode', action='store_true',
                        help='put code in debuging mode.')
    parser.add_argument('--adaptive_buffer', action='store_true',
                        help='activate adapitive buffer')
    parser.add_argument('--buffer_step_size', default=20)

    ARGS = parser.parse_args()

    # -------setup seed-----------
    random.seed(ARGS.seed_value)
    torch.manual_seed(ARGS.seed_value)
    np.random.seed(ARGS.seed_value)
    # ----------------------------

    print(ARGS)
    main()
    # evaluate()

# python train.py --num_episodes 1000 --batch_size 64 --render_env 10 --num_hidden 64 --lr 5e-4 --discount_factor 0.8 --replay NaiveReplayMemory --env CartPole-v1 --buffer 10000 --pmethod prop --TAU 0.1
# python train.py --num_episodes 1000 --batch_size 64 --render_env 10 --num_hidden 64 --lr 5e-4 --discount_factor 0.99 --replay NaiveReplayMemory --env LunarLander-v2 --buffer 100000 --pmethod prop --TAU 0.1
# python train.py --env MountainCar-v0 --lr 5e-4 --render_env 10 --discount_factor 0.99 --TAU 0.1 --buffer 10000 --replay CombinedReplayMemory
