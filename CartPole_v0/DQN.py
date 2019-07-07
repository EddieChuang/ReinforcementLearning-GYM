import gym
import numpy as np
import random
from collections import deque
import tensorflow as tf

game = 'CartPole'

env = gym.make('{}-v0'.format(game))
env = gym.wrappers.Monitor(env, 'log/QL/{}'.format(game), force=True)

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

        h = tf.keras.layers.Dense(64, activation='relu', name=self.name + '_hidden0')(self.state)  # (batch_size, 24)
        h = tf.keras.layers.Dense(128, activation='relu', name=self.name + '_hidden1')(h)  # (batch_size, 24)
        self.q = tf.keras.layers.Dense(n_action, name=self.name + '_output')(h)  # (batch_size, 2)

        self.loss = tf.losses.mean_squared_error(labels=self.value, predictions=self.q)  # (batch_size, 2)
        self.loss = tf.reduce_mean(self.loss)  # ()

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
    
    def predict(self, states):
        '''
        Parameters:
          states: np.array of states. shape is (None, 4)
        '''

        fetches = [self.q]
        q = sess.run(self.q, {self.state: states})
        
        return q[0]


    def get_variables(self):

        return [var for var in tf.trainable_variables() if self.name in var.name]
    
    def save(self):
        print('Save model to {}'.format('models/QL/model'))
        saver = tf.train.Saver()
        saver.save(sess, 'models/QL/model')

    def restore(self):
        print('Restore model from {}'.format('models/QL/model'))
        saver = tf.train.Saver()
        saver.restore(sess, 'models/QL/model')
    

class ReplayBuffer:
    def __init__(self, max_buffer_size):
        self.max_buffer_size = max_buffer_size
        self.experience = deque()
    
    def add(self, exp):
        if self.size() > self.max_buffer_size:
            self.experience.popleft()
        self.experience.append(exp)
    
    def sample(self, batch_size):
        return random.sample(self.experience, batch_size)
    
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

    x_train, y_train = [], []

    for (state, action, reward, next_state, done) in training_data:
        state = preprocess_state(state)
        next_state = preprocess_state(next_state)

        value = training_network.predict(state)  # (1, 2)
        value[action] = reward
        if not done:
            value[action] += 0.95 * np.max(target_network.predict(next_state))
        
        x_train.append(state)
        y_train.append(value)
    
    x_train = np.array(x_train).reshape((-1, n_input))   # (32, 4)
    y_train = np.array(y_train).reshape((-1, n_action))  # (32, 2)
    
    loss = training_network.fit(x_train, y_train)
    return loss




def train():
    max_buffer_size = 100000
    num_episodes = 3000
    batch_size = 32
    
    epsilon = 1
    epsilon_min = 0.1
    epsilon_decay = 0.999

    training_network = QNetwork(name='training')
    target_network = QNetwork(name='target')
    
    sess.run(tf.global_variables_initializer())
    update_target_network(training_network, target_network)
   

    replaybuffer = ReplayBuffer(max_buffer_size)


    global_step = 0
    for e in range(num_episodes):
        state = env.reset()
        done = False
        score, avg_loss, avg_maxq, count = 0, 0, 0, 0
        while not done:
            action = choose_action(env, training_network, state, epsilon)
            next_state, reward, done, info = env.step(action)

            replaybuffer.add((state, action, reward, next_state, done))
            state = next_state

            score += 1
            global_step += 1
            count += 1
        
            if len(replaybuffer.experience) > batch_size:
                # print('Updating Training Network ...\n')
                training_data = replaybuffer.sample(batch_size)
                avg_loss += update_training_network(training_network, target_network, training_data)

            avg_maxq += np.max(training_network.predict(preprocess_state(state)))

        update_target_network(training_network, target_network)
        
        if (e + 1) % 100 == 0:
            training_network.save()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        avg_loss /= count
        avg_maxq /= count
        print('Step: {} | Score: {} | MAXQ: {} | AVG_Loss: {} | Epsilon: {}'.format(global_step, score, avg_maxq, avg_loss, epsilon))

    print('Global Step: {}'.format(global_step))

def evaluation():
    training_network = QNetwork(name='training')
    training_network.restore()
    
    score = 0
    state = env.reset()
    while True:
        state = preprocess_state(state)
        action = np.argmax(training_network.predict(state))
        state, reward, done, info = env.step(action)
        score += reward

        if done:
            print('Score: {}'.format(score))
            break


def main():
    # train()
    evaluation()

   


if __name__ == '__main__':
    main()