import dataset3
import dataset4
import dataset5
import dataset6
from env import Environment
from dqn import DQN
import config
from tqdm import tqdm
import os
import torch
import sys

dataset = dataset5


def main():
    config.device_name = "cuda:{}".format(sys.argv[1])
    config.device = torch.device(config.device_name)
    multi_train(dataset.file_name, 0, 25, truncate_file=True)


def output_parameters():
    print("Gap penalty: {}".format(config.GAP_PENALTY))
    print("Mismatch penalty: {}".format(config.MISMATCH_PENALTY))
    print("Match reward: {}".format(config.MATCH_REWARD))
    print("Episode: {}".format(config.max_episode))
    print("Batch size: {}".format(config.batch_size))
    print("Replay memory size: {}".format(config.replay_memory_size))
    print("Alpha: {}".format(config.alpha))
    print("Epsilon: {}".format(config.epsilon))
    print("Gamma: {}".format(config.gamma))
    print("Delta: {}".format(config.delta))
    print("Decrement iteration: {}".format(config.decrement_iteration))
    print("Update iteration: {}".format(config.update_iteration))
    print("Device: {}".format(config.device_name))


def multi_train(tag="", start=0, end=-1, truncate_file=False):
    output_parameters()
    print("Dataset number: {}".format(len(dataset.datasets)))

    report_file_name = os.path.join(config.report_path, "{}.rpt".format(tag))

    if truncate_file:
        with open(report_file_name, 'w') as _:
            _.truncate()

    for index, name in enumerate(dataset.datasets[start:end if end != -1 else len(dataset.datasets)], start):
        if not hasattr(dataset, name):
            continue
        seqs = getattr(dataset, name)

        env = Environment(seqs)
        agent = DQN(env.action_number, env.row, env.max_len, env.max_len * env.max_reward)
        p = tqdm(range(config.max_episode))
        p.set_description(name)

        for _ in p:
            state = env.reset()
            while True:
                action = agent.select(state)
                reward, next_state, done = env.step(action)
                agent.replay_memory.push((state, next_state, action, reward, done))
                agent.update()
                if done == 0:
                    break
                state = next_state
            agent.update_epsilon()

        state = env.reset()

        while True:
            action = agent.predict(state)
            _, next_state, done = env.step(action)
            state = next_state
            if 0 == done:
                break

        env.padding()
        report = "{}\n{}\n{}\n{}\n{}\n{}\n{}\n\n".format("NO: {}".format(name),
                                                         "AL: {}".format(len(env.aligned[0])),
                                                         "SP: {}".format(env.calc_score()),
                                                         "EM: {}".format(env.calc_exact_matched()),
                                                         "CS: {}".format(
                                                             env.calc_exact_matched() / len(env.aligned[0])),
                                                         "QTY: {}".format(len(env.aligned)),
                                                         "#\n{}".format(env.get_alignment()))

        with open(os.path.join(config.report_path, "{}.rpt".format(tag)), 'a+') as report_file:
            report_file.write(report)


def train(index):
    output_parameters()

    assert hasattr(dataset, "dataset_{}".format(index)), "No such data called {}".format("dataset_{}".format(index))
    data = getattr(dataset, "dataset_{}".format(index))

    print("{}: dataset_{}: {}".format(dataset.file_name, index, data))

    env = Environment(data)
    agent = DQN(env.action_number, env.row, env.max_len, env.max_len * env.max_reward)
    p = tqdm(range(config.max_episode))

    for _ in p:
        state = env.reset()
        while True:
            action = agent.select(state)
            reward, next_state, done = env.step(action)
            agent.replay_memory.push((state, next_state, action, reward, done))
            agent.update()
            if done == 0:
                break
            state = next_state
        agent.update_epsilon()

    # Predict
    state = env.reset()
    while True:
        action = agent.predict(state)
        _, next_state, done = env.step(action)
        state = next_state
        if 0 == done:
            break

    env.padding()
    print("**********dataset: {} **********\n".format(data))
    print("total length : {}".format(len(env.aligned[0])))
    print("sp score     : {}".format(env.calc_score()))
    print("exact matched: {}".format(env.calc_exact_matched()))
    print("column score : {}".format(env.calc_exact_matched() / len(env.aligned[0])))
    print("alignment: \n{}".format(env.get_alignment()))
    print("********************************\n")


if __name__ == "__main__":
    main()
