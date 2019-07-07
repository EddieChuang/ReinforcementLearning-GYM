import gym
import numpy as np
import random
from collections import deque
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

game = 'MountainCar'

env = gym.make('{}-v0'.format(game))
# env.seed(555)
# env = gym.wrappers.Monitor(env, 'log/QL/{}'.format(game), force=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

n_action = env.action_space.n
n_input = env.observation_space.shape[0]

print('n_action: ', n_action)
print('n_input: ', n_input)
# print(env.get_action_meanings())

class QNetwork:
    def __init__(self, name='name'):
        self.name = name
        self._build()
        
    
    def _build(self):
        
        self.state = tf.placeholder(tf.float32, shape=(None, n_input), name=self.name + '_state')
        self.value = tf.placeholder(tf.float32, shape=(None, n_action), name=self.name + '_value')

        h = tf.keras.layers.Dense(20, activation='relu', name=self.name + '_hidden0')(self.state)  # (batch_size, 24)
        h = tf.keras.layers.Dense(25, activation='relu', name=self.name + '_hidden1')(h)  # (batch_size, 24)
        self.q = tf.keras.layers.Dense(n_action, name=self.name + '_output')(h)  # (batch_size, 2)

        self.loss = tf.losses.mean_squared_error(labels=self.value, predictions=self.q)  # ()
        # self.loss = tf.reduce_mean(self.loss)  # ()

        self.train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def fit(self, x_train, y_train):
        '''
        Parameters:
          x_train: np.array of states. shape is (None, 4)
          y_train: np.array of qvalue. shape is (None, 2)
        '''

        fetches = [self.loss, self.train_op]
        feed_dict = {self.state: x_train, self.value: y_train}
        loss, _ = sess.run(fetches, feed_dict)

        return loss
        # print('Loss: {}'.format(loss))
    
    def predict(self, states):
        '''
        Parameters:
          states: np.array of states. shape is (None, n_input)
        '''

        fetches = [self.q]
        q = sess.run(fetches, {self.state: states})
        
        return q[0]

    def get_variables(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]
    
    def save(self):
        print('Save model to {}'.format('models/model'))
        saver = tf.train.Saver()
        saver.save(sess, 'models/model')

    def restore(self):
        print('Restore model from {}'.format('models/model'))
        saver = tf.train.Saver()
        saver.restore(sess, 'models/model')
    

class ReplayBuffer:
    def __init__(self, max_buffer_size):
        self.max_buffer_size = max_buffer_size
        self.experience = deque()
    
    def add(self, exp):
        if self.size() > self.max_buffer_size:
            self.experience.popleft()
        self.experience.append(exp)
    
    def sample(self, batch_size):
        return np.array(random.sample(self.experience, batch_size))
    
    def size(self):
        return len(self.experience)


def preprocess_state(state):
    return np.reshape(state, (1, n_input))

def choose_action(env, network, state, epsilon):
    if np.random.random() <= epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(network.predict(preprocess_state(state)))

def update_target_network(training_network, target_network):
    
    training_vars = training_network.get_variables()
    target_vars = target_network.get_variables()

    assign_ops = []
    for training_var, target_var in zip(training_vars, target_vars):
        assign_ops.append(target_var.assign(training_var))

    
    sess.run(assign_ops)
    
def update_training_network(training_network, target_network, training_data):
    
    states  = np.array([state[0] for state in training_data[:, 0]])      # (batch_size, n_input)
    actions = np.array(training_data[:, 1]).astype(np.int32)             # (batch_size, n_action)
    rewards = np.array(training_data[:, 2]).astype(np.float)             # (batch_size, 1)
    next_states = np.array([state[0] for state in training_data[:, 3]])  # (batch_size, n_input)
    dones = training_data[:, 4].astype(np.int32)                         # (batch_size, 1)
    
    indice = np.arange(len(states))
    values = training_network.predict(states)                            # (batch_size, n_action)
    values[indice, actions] = rewards + 0.95 * np.max(target_network.predict(next_states), axis=1) * (1 - dones)
    # print(values)
    loss = training_network.fit(states, values)
    return loss

def get_reward(position):
    if position >= 0.5:
        return 10
    elif position > -0.4:
        return (1 + position) ** 4
    return 0
    

def train():
    max_buffer_size = 100000
    max_step = 20000
    batch_size = 64
    
    epsilon_max = 1
    epsilon_min = 0.0
    epsilon = epsilon_max
    aneal_step = 1000

    training_network = QNetwork(name='training')
    target_network = QNetwork(name='target')
    
    sess.run(tf.global_variables_initializer())
    update_target_network(training_network, target_network)

    replaybuffer = ReplayBuffer(max_buffer_size)

    global_step = 0
    while global_step < max_step:
        state = preprocess_state(env.reset())  # (1, n_input)
        done = False


        score, avg_reward, avg_loss, avg_maxq, count = 0, 0, 0, 0, 0
        while not done:
            env.render()
            action = choose_action(env, training_network, state, epsilon)

            next_state, reward, done, info = env.step(action)
            next_state = preprocess_state(next_state)    # (1, n_input)
            reward = get_reward(next_state[0][0])
            score += reward

            replaybuffer.add((state, action, reward, next_state, done))
            state = next_state

            count += 1
            global_step += 1
            if epsilon >  epsilon_min:
                epsilon -= (epsilon_max - epsilon_min) / aneal_step

            if replaybuffer.size() > batch_size:
                training_data = replaybuffer.sample(batch_size)
                avg_loss += update_training_network(training_network, target_network, training_data)
            
            if global_step % 10000 == 0:
                training_network.save()

            avg_reward += reward
            avg_maxq += np.max(training_network.predict(state))

            if global_step % 50 == 0:
                update_target_network(training_network, target_network)
            
            
        avg_reward /= count
        avg_loss /= count
        avg_maxq /= count
        score /= count
        print('Step: {} | Score: {} | MAXQ: {} | AVG_Reward: {} | AVG_Loss: {} | Epsilon: {}'.format(global_step, score, avg_maxq, avg_reward,  avg_loss, epsilon))
    print('Step: {} | Score: {} | Epsilon: {}'.format(global_step, score, epsilon))


    training_network.save()
    print('Global Step: {}'.format(global_step))


def evaluation():
    training_network = QNetwork(name='training')
    training_network.restore()
    
    score = 0
    state = env.reset()

    while True:
        env.render()
        state = preprocess_state(state)
        action = np.argmax(training_network.predict(state))
        
        state, reward, done, info = env.step(action)

        score += reward

        if done:
            print('Score: {}'.format(score))
            break

def main():
    train()    
    # evaluation()

    
    



if __name__ == '__main__':
    main()