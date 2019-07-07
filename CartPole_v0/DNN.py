import tensorflow as tf
import gym, math
import numpy as np

env = gym.make('CartPole-v0')

def create_training_gameaction(num_train_data=5000):
    train_data = []  # [observation, action]

    threshold_score = 70
    game_step_limit = 500    
    while len(train_data) < num_train_data:
        env.reset()
        game_actions = []
        prev_observation = np.array([])

        score = 0
        for _ in range(game_step_limit):
            env.render()
            action = env.action_space.sample()
            if prev_observation.any():
                game_actions.append([prev_observation.tolist(), action])
            
            prev_observation, reward, done, info = env.step(action)
            score += reward
            
            if done:
                break

        if score >= threshold_score:
            train_data.extend(game_actions)
            print('Created {} trainging data'.format(len(train_data)))

    np.save('data/train_data_gameaction-{}.npy'.format(num_train_data), train_data)
    return np.array(train_data)

def next_batch(train_data, batch_size=32):

    x, y = [_ for _ in train_data[:, 0]], [_ for _ in train_data[:, 1]]
    n_batch = math.ceil(len(x) / batch_size)

    for i in range(n_batch):
        offset = i * batch_size
        yield x[offset: offset + batch_size], y[offset: offset + batch_size]

def shuffle(train_data):
    indice = np.arange(len(train_data))
    np.random.shuffle(indice)
    return train_data[indice]

def create_model():

    num_actions = env.action_space.n
    num_inputs = env.observation_space.shape[0]
    hidden_units = [32, 64]

    x = tf.placeholder(tf.float32, shape=(None, num_inputs))
    y = tf.placeholder(tf.int32, shape=(None, ))

    h = x
    for units in hidden_units:
        h = tf.keras.layers.Dense(units=units, activation='relu')(h)     # (batch_size, hidden_units)
        h = tf.keras.layers.Dropout(0.5)(h)

    o = tf.keras.layers.Dense(units=num_actions, activation=None)(h)     # (batch_size, num_actions)
    probabilities = tf.keras.activations.softmax(o)                      # (batch_size, num_actions)
    y_onehot = tf.one_hot(y, depth=num_actions)                          # (batch_size, num_actions)

    cross_entropy = tf.losses.softmax_cross_entropy(y_onehot, o)         # (batch_size, )
    loss = tf.reduce_mean(cross_entropy)  # () 
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    
    return x, y, loss, optimizer, probabilities


if __name__ == '__main__':

    num_train_data = 20000
    ########## Create Training Data ##########
    # create_training_gameaction(num_train_data=num_train_data)

    ########## Load Training Data ##########
    train_data = np.load('data/train_data_gameaction-{}.npy'.format(num_train_data))

    ########## Create Model ##########
    model_x, model_y, model_loss, model_optimizer, model_probabilities = create_model()


    ########## Train Model ##########
    epoch_size = 10
    batch_size = 32
    initializer = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(initializer)
        fetches = [model_loss, model_optimizer, model_probabilities]
        
        for i in range(epoch_size):
            train_data = shuffle(train_data)
            avg_loss = 0
            for batch_x, batch_y in next_batch(train_data, batch_size):
                feed_dict = {model_x: batch_x, model_y: batch_y}
                loss, _, probabilities = sess.run(fetches, feed_dict)
                avg_loss += loss
            print(avg_loss / (int(len(train_data) / batch_size)))
            
        saver = tf.train.Saver()
        saver.save(sess, 'models/DNN/dnn')



    ########## Use Trained Model to Play CartPole-v0 ##########
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, 'models/DNN/dnn')
        env.reset()

        # observation, reward, done, info = env.step(env.action_space.sample())
        observation = env.reset()
        score = 0
        for _ in range(1000):
            env.render()
            
            probabilities = sess.run([model_probabilities], {model_x: observation.reshape((-1, 4))})

            action = np.argmax(probabilities[0])
            observation, reward, done, info = env.step(action)
            score += reward
            if done:
                print('Gave Over. Your Score is ', score)
                break

    env.close()


# observation: [position of cart, velocity of cart, angle of pole, rotation rate of pole]