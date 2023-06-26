import platform
import numpy as np
import tkinter as tk
import tkinter.font as tf
import copy
import config
from itertools import combinations

colors = ["#FFFFFF", "#5CB85C", "#5BC0DE", "#F0AD4E", "#D9534F", "#808080"]
nucleotides_map = {'A': 1, 'T': 2, 'C': 3, 'G': 4, 'a': 1, 't': 2, 'c': 3, 'g': 4, '-': 5}
nucleotides = ['A', 'T', 'C', 'G', '-']


class Environment:
    def __init__(self, data,
                 nucleotide_size=50, text_size=25,
                 show_nucleotide_name=True):
        self.data = [[nucleotides_map[data[i][j]] for j in range(len(data[i]))] for i in range(len(data))]
        self.row = len(data)
        self.max_len = max([len(data[i]) for i in range(len(data))])
        self.show_nucleotide_name = show_nucleotide_name
        self.nucleotide_size = nucleotide_size
        self.max_window_width = 1800
        self.text_size = text_size

        self.action_number = 2 ** self.row - 1

        self.max_reward = self.row * (self.row - 1) / 2 * config.MATCH_REWARD

        self.aligned = [[] for _ in range(self.row)]
        self.not_aligned = copy.deepcopy(self.data)

        if platform.system() == "Windows":
            self.window = tk.Tk()
            self.__init_size()
            self.__init_window()
            self.__init_canvas()

    def __action_combination(self):
        res = []
        for i in range(self.row + 1):
            combs = list(combinations(range(self.row), i))

            for j in combs:
                a = np.zeros(self.row)
                for k in j:
                    a[k] = 1
                res.append(a)

        res.pop()

        return res

    def __init_size(self):
        self.window_default_width = (self.max_len + 2) * self.nucleotide_size if \
            (self.max_len + 2) * self.nucleotide_size < self.max_window_width else self.max_window_width

        self.window_default_height = self.nucleotide_size * (2 * self.row + 2) + 40
        self.nucleotide_font = tf.Font(family="bold", size=self.text_size * 2 // 3, weight=tf.BOLD)

    def __init_window(self):
        self.window.maxsize(self.window_default_width, self.window_default_height)
        self.window.minsize(self.window_default_width, self.window_default_height)
        self.window.title("Multiple Sequence Alignment")

    def __init_canvas(self):
        self.frame = tk.Frame(self.window, width=self.window_default_width,
                              height=self.window_default_height)
        self.frame.pack()

        self.canvas = tk.Canvas(self.frame, width=self.nucleotide_size * (self.max_len + 1),
                                height=self.nucleotide_size * (self.row + 1),
                                scrollregion=(
                                    0, 0, self.nucleotide_size * (len(self.aligned[0]) + 1),
                                    self.nucleotide_size * (self.row + 1)))

        self.scroll = tk.Scrollbar(self.frame, orient="horizontal", width=20)
        self.scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.scroll.config(command=self.canvas.xview)
        self.canvas.config(xscrollcommand=self.scroll.set, width=self.max_window_width,
                           height=self.window_default_height)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def __get_current_state(self):
        state = []
        for i in range(self.row):
            state.extend((self.not_aligned[i][j] if j < len(self.not_aligned[i]) else 5)
                         for j in range(len(self.not_aligned[i]) + 1))

        state.extend([0 for _ in range(self.row * (self.max_len + 1) - len(state))])
        return state

    def __calc_reward(self):
        score = 0
        tail = len(self.aligned[0]) - 1
        for j in range(self.row):
            for k in range(j + 1, self.row):
                if self.aligned[j][tail] == 5 or self.aligned[k][tail] == 5:
                    score += config.GAP_PENALTY
                elif self.aligned[j][tail] == self.aligned[k][tail]:
                    score += config.MATCH_REWARD
                elif self.aligned[j][tail] != self.aligned[k][tail]:
                    score += config.MISMATCH_PENALTY

        return score

    def __show_alignment(self):
        self.canvas.delete(tk.ALL)
        rx_start = self.nucleotide_size // 2
        ry_start = self.nucleotide_size // 2
        nx_start = self.nucleotide_size
        ny_start = self.nucleotide_size
        for i in range(self.row):
            for j in range(len(self.aligned[i])):
                self.canvas.create_rectangle(j * self.nucleotide_size + rx_start,
                                             i * self.nucleotide_size + ry_start,
                                             (j + 1) * self.nucleotide_size + rx_start,
                                             (i + 1) * self.nucleotide_size + ry_start,
                                             fill=colors[self.aligned[i][j]], outline="#757575")
                if self.show_nucleotide_name:
                    self.canvas.create_text(j * self.nucleotide_size + nx_start,
                                            i * self.nucleotide_size + ny_start,
                                            text=nucleotides[self.aligned[i][j] - 1],
                                            font=self.nucleotide_font,
                                            fill="white")

        ry_start += (self.row + 1) * self.nucleotide_size
        ny_start += (self.row + 1) * self.nucleotide_size
        for i in range(self.row):
            for j in range(len(self.not_aligned[i])):
                self.canvas.create_rectangle(j * self.nucleotide_size + rx_start,
                                             i * self.nucleotide_size + ry_start,
                                             (j + 1) * self.nucleotide_size + rx_start,
                                             (i + 1) * self.nucleotide_size + ry_start,
                                             fill=colors[self.not_aligned[i][j]], outline="#757575")
                if self.show_nucleotide_name:
                    self.canvas.create_text(j * self.nucleotide_size + nx_start,
                                            i * self.nucleotide_size + ny_start,
                                            text=nucleotides[self.not_aligned[i][j] - 1],
                                            font=self.nucleotide_font,
                                            fill="white")

        scroll_width = len(self.aligned[0]) if len(self.aligned[0]) > len(self.not_aligned[0]) else \
            len(self.not_aligned[0])
        self.canvas['scrollregion'] = (0, 0, self.nucleotide_size * (scroll_width + 1),
                                       self.nucleotide_size * (self.row + 1))
        self.window.update()

    def reset(self):
        self.aligned = [[] for _ in range(self.row)]
        self.not_aligned = copy.deepcopy(self.data)
        return self.__get_current_state()

    def step(self, action):
        for bit in range(self.row):
            if 0 == (action >> bit) & 0x1 and 0 == len(self.not_aligned[bit]):
                return -self.max_reward, self.__get_current_state(), 0

        total_len = 0
        for bit in range(self.row):
            if 0 == (action >> bit) & 0x1:
                self.aligned[bit].append(self.not_aligned[bit][0])
                self.not_aligned[bit].pop(0)
            else:
                self.aligned[bit].append(5)
            total_len += len(self.not_aligned[bit])

        return self.__calc_reward(), self.__get_current_state(), 1 if total_len > 0 else 0

    def calc_score(self):
        score = 0
        for i in range(len(self.aligned[0])):
            for j in range(self.row):
                for k in range(j + 1, self.row):
                    if self.aligned[j][i] == 5 or self.aligned[k][i] == 5:
                        score += config.GAP_PENALTY
                    elif self.aligned[j][i] == self.aligned[k][i]:
                        score += config.MATCH_REWARD
                    elif self.aligned[j][i] != self.aligned[k][i]:
                        score += config.MISMATCH_PENALTY

        return score

    def calc_exact_matched(self):
        score = 0

        for i in range(len(self.aligned[0])):
            n = self.aligned[0][i]
            flag = True
            for j in range(1, self.row):
                if n != self.aligned[j][i]:
                    flag = False
                    break
            if flag:
                score += 1

        return score

    def set_alignment(self, seqs):
        self.aligned = [[nucleotides_map[seqs[i][j]] for j in range(len(seqs[i]))] for i in range(len(seqs))]
        self.not_aligned = [[] for _ in range(len(self.data))]

    def render(self):
        if platform.system() == "Windows":
            self.__show_alignment()

    def get_alignment(self):
        alignment = ""
        for i in range(len(self.aligned)):
            alignment += ''.join([nucleotides[self.aligned[i][j] - 1] for j in range(len(self.aligned[i]))]) + '\n'

        return alignment.rstrip()

    def padding(self):
        max_length = 0
        for i in range(len(self.not_aligned)):
            max_length = max(max_length, len(self.not_aligned[i]))

        for i in range(len(self.not_aligned)):
            self.aligned[i].extend(self.not_aligned[i])
            self.aligned[i].extend([5 for _ in range(max_length - len(self.not_aligned[i]))])
            self.not_aligned[i].clear()
