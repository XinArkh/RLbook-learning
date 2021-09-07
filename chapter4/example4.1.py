'''
The idea of Gridworld class refers to:
[MatthewJA/Inverse-Reinforcement-Learning](https://github.com/MatthewJA/Inverse-Reinforcement-Learning)
'''

import numpy as np


class Gridworld:
    """
    Gridworld MDP.
    """

    def __init__(self, grid_size, gamma):
        """
        grid_size: Grid size. int or 2D tuple/list.
        gamma: MDP discount. float.
        -> Gridworld
        """

        self.grid_size = grid_size
        if type(grid_size) is int:
            self.grid_rows = self.grid_columns = grid_size
        elif (type(grid_size) is tuple or type(grid_size) is list) and \
            len(grid_size) == 2:
            self.grid_rows, self.grid_columns = grid_size
        else:
            raise Exception('Invalid grid_size: {}'.format(grid_size))

        self.actions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        self.n_actions = len(self.actions)
        self.n_states = self.grid_rows * self.grid_columns
        self.gamma = gamma
        self.transition_probability = np.array(
            [[[self._transition_probability(i, j, k)
               for k in range(self.n_states)]
              for j in range(self.n_actions)]
             for i in range(self.n_states)])
        self.policy = np.ones((self.n_states, self.n_actions)) / 4
        self.values = np.zeros(self.n_states)

    def __str__(self):
        return "Gridworld({}, {})".format(self.grid_size, self.gamma)

    def reward(self, i_state):
        if i_state not in {0, self.n_states-1}:
            return -1
        else:
            return 0

    def int_to_point(self, i):
        """
        Convert a state int into the corresponding coordinate.

        i: State int.
        -> (x, y) int tuple.
        """

        return (i % self.grid_columns, i // self.grid_columns)

    def point_to_int(self, p):
        """
        Convert a coordinate into the corresponding state int.

        p: (x, y) tuple.
        -> State int.
        """

        return p[0] + p[1] * self.grid_columns

    def neighbouring(self, i, k):
        """
        Get whether two points neighbour each other. Also returns true if they
        are the same point.

        i: (x, y) int tuple.
        k: (x, y) int tuple.
        -> bool.
        """

        return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 1

    def _transition_probability(self, i, j, k):
        """
        Get the probability of transitioning from state i to state k given
        action j.

        i: State int.
        j: Action int.
        k: State int.
        -> p(s_k | s_i, a_j)
        """

        xi, yi = self.int_to_point(i)
        xj, yj = self.actions[j]
        xk, yk = self.int_to_point(k)

        if not self.neighbouring((xi, yi), (xk, yk)):
            return 0.0

        # Is k the intended state to move to?
        if (xi + xj, yi + yj) == (xk, yk):
            return 1.0

        # If these are not the same point, then we can not move there.
        if (xi, yi) != (xk, yk):
            return 0.0

        # If these are the same point, we can only move here by moving
        # off the grid. Are we on a corner or not?
        if (xi, yi) in {(0, 0), (self.grid_columns-1, self.grid_rows-1),
                        (0, self.grid_rows-1), (self.grid_columns-1, 0)}:
            # Corner.
            # Can move off the edge in two directions.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_columns and
                    0 <= yi + yj < self.grid_rows):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here.
                return 1.0
            else:
                return 0.0
        else:
            # Not a corner. Is it an edge?
            if (xi not in {0, self.grid_columns-1} and
                yi not in {0, self.grid_rows-1}):
                # Not an edge.
                return 0.0
            # Edge.
            # Can only move off the edge in one direction.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_columns and
                    0 <= yi + yj < self.grid_rows):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here.
                return 1.0
            else:
                return 0.0

    def policy_evaluation(self, theta=0.0001, verbose=False):
        print('Policy evaluation: start policy evaluation...')

        Delta = float('inf')
        t = 0
        while Delta >= theta:
            t += 1
            Delta = 0
            for i_state in range(self.n_states):
                if i_state in {0, self.n_states-1}:
                    continue
                v_old = self.values[i_state]
                v_tmp = 0
                for i_action in range(self.n_actions):
                    v_tmp += np.sum(self.policy[i_state, i_action] * \
                        self.transition_probability[i_state, i_action] * \
                        (self.reward(i_state) + self.gamma * self.values))
                self.values[i_state] = v_tmp
                Delta = np.max([Delta, np.abs(v_old - v_tmp)])
            
            if verbose:
                print('--'*10 + ' policy evaluation iter {:2d} '.format(t) + '--'*10)
                print('value matrix:')
                print(self.values.reshape((self.grid_rows, self.grid_columns)))
        
        print('Policy evaluation: value converges in {} steps.'.format(t))
        return self.values

    def policy_iteration(self, theta=0.0001, epsilon=0.0001, verbose=False):
        print('Policy iteration: start policy iteration...')

        policy_stable = False
        t = 0
        while not policy_stable:
            # policy evaluation
            self.policy_evaluation(theta=theta, verbose=verbose)
            
            # policy improvement
            t += 1
            policy_stable = True
            for i_state in range(self.n_states):
                if i_state in {0, self.n_states-1}:
                    continue
                pi_old = self.policy[i_state].copy()
                v_max = -float('inf')
                v_a = np.zeros(self.n_actions)
                for i_action in range(self.n_actions):
                    v_a[i_action] = np.sum(self.transition_probability[i_state, i_action] * \
                                           (self.reward(i_state) + self.gamma * self.values))
                a_max = np.max(v_a)
                self.policy[i_state] = np.array(v_a >= a_max - epsilon, dtype=float) / \
                                        np.sum(np.array(v_a >= a_max - epsilon, dtype=float))
                if not np.all(self.policy[i_state][pi_old > 0]):
                    policy_stable = False
                
            if verbose:
                print('\n' + '**'*10 + ' policy improvement iter {:2d} '.format(t) + '**'*10)
                print('polocy matrix:')
                print(self.policy, '\n')

        print('Policy iteration: stable policy found!')
        if verbose:
            print('value matrix:')
            print(self.values.reshape((self.grid_rows, self.grid_columns)))
            print('polocy matrix:')
            print(self.policy)

        return self.values, self.policy


if __name__ == '__main__':
    grid_size = 4
    gamma = 1
    gw = Gridworld(grid_size, gamma)
    # gw.policy_evaluation(verbose=True)
    gw.policy_iteration(verbose=True)