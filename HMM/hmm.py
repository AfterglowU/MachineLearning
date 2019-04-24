'''
@Author: Wu Yuhui
@Time  : 2019.04.24
'''
import numpy as np

class HMM():
    def __init__(self):
        # 状态空间, N = 3
        # Y   =   { 0       1       2       }
        # weather : sunny   cloudy  rainy
        self.N = 3
        # 观测值集合 (输出空间), M = 3
        # X   =   { 0       1       2       }
        # stone is: hot     cold    wet
        self.M = 3
        self.Pi = np.array( [30/47, 9/47, 8/47] )
        self.A = np.array([ [0.8, 0.1, 0.1], 
                            [0.3, 0.4, 0.3],
                            [0.4, 0.2, 0.4] ])
        self.B = np.array([ [0.8, 0.2, 0.1],
                            [0.1, 0.5, 0.2],
                            [0.1, 0.3, 0.7] ]).T

    def forward(self, observe_sequence):
        # N * len(observe_sequence)
        dp = np.array([[0.0 for t in range(len(observe_sequence))] for i in range(self.N)])
        # Init
        for i in range(self.N):
            dp[i][0] = self.Pi[i] * self.B[i][observe_sequence[0]]
        # Bottom-up
        for t in range(1, len(observe_sequence)):
            for i in range(self.N):
                dp[i][t] = sum( [dp[k][t-1] * self.A[k][i] * self.B[i][observe_sequence[t]]
                                for k in range(self.N)] )
        # Add the last column up
        p = sum([ dp[i][-1] for i in range(self.N) ])
        return p
    
    def viterbi(self, observe_sequence):
        # Both N * len(observe_sequence)
        dp = np.array([[0.0 for t in range(len(observe_sequence))] for i in range(self.N)])
        path = np.array([[0 for t in range(len(observe_sequence))] for i in range(self.N)])
        # Init
        for i in range(self.N):
            dp[i][0] = self.Pi[i] * self.B[i][observe_sequence[0]]
            path[i][0] = 0
        # Bottom-up
        for t in range(1, len(observe_sequence)):
            for i in range(self.N):
                tmp = [dp[k][t-1] * self.A[k][i] for k in range(self.N)]
                path[i][t] = tmp.index(max(tmp))
                dp[i][t] = tmp[path[i][t]] * self.B[i][observe_sequence[t]]
        # Backtrack to get state_sequence
        tmp = [dp[i][-1] for i in range(self.N)]
        end_state = tmp.index(max(tmp))
        state_sequence = np.array([0 for i in range(len(observe_sequence))])
        state_sequence[-1] = end_state
        for t in range(len(observe_sequence) - 1):
            state_sequence[-(t+2)] = path[state_sequence[-(t+1)]][-(t+1)]
        return state_sequence
        

if __name__ == '__main__':
    hmm = HMM()
    observe = input("请输入观测序列(用\',\'分隔):") # 0,0,1,2,1,2,1,0
    observe_sequence = [int(t) for t in observe.split(",")]
    # test forward algo
    p = hmm.forward(observe_sequence)
    print("该观测序列由该模型产生的概率为:" + str(p))
    # test viterbi algo
    state_sequence = hmm.viterbi(observe_sequence)
    state_sequence_read = []
    for state in state_sequence:
        if state == 0:
            state_sequence_read.append("晴")
        elif state == 1:
            state_sequence_read.append("阴")
        elif state == 2:
            state_sequence_read.append("雨")
        else:
            pass
    print("该观测序列对应的最优状态序列为:" + str(state_sequence_read))
