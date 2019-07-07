import tensorflow as tf
import gym
import numpy as np

env = gym.make('CartPole-v0')
num_actions = env.action_space.n
num_inputs = env.observation_space.shape[0]

def create_model():

    
    hidden_units = [4]

    x = tf.placeholder(tf.float32, shape=(None, num_inputs))
    y = tf.placeholder(tf.int32, shape=(None, ))

    h = x
    for units in hidden_units:
        h = tf.keras.layers.Dense(units=units, activation='relu')(h)     # (batch_size, hidden_units)
        # h = tf.keras.layers.Dropout(0.5)(h)

    o = tf.keras.layers.Dense(units=num_actions, activation=None)(h)     # (batch_size, num_actions)
    probabilities = tf.keras.activations.softmax(o)                      # (batch_size, num_actions)

    action = tf.multinomial(probabilities, num_samples=1)                # (batch_size, 1)

    y_onehot = tf.one_hot(y, depth=num_actions)                     # (batch_size, num_actions)

    cross_entropy = tf.losses.softmax_cross_entropy(y_onehot, o)         # (batch_size, )
    loss = tf.reduce_mean(cross_entropy)                                 # ()

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    gradients_and_variables = optimizer.compute_gradients(loss)

    gradients = []
    gradient_placeholders = []
    grads_and_vars_feed = []
    for gradient, variable in gradients_and_variables:
        gradients.append(gradient)
        gradient_placeholder = tf.placeholder(tf.float32, shape=gradient.get_shape())
        gradient_placeholders.append(gradient_placeholder)
        grads_and_vars_feed.append((gradient_placeholder, variable))


    training_op = optimizer.apply_gradients(grads_and_vars_feed)

    return x, y, action, gradients, loss, gradient_placeholders, training_op


def helper_discount_rewards(rewards, discount_rate):
    '''
    Parameters:
    rewards: (game_steps, ). reward for each round

    Return:
    discounted_rewards (game_steps, ). discounted reward for each round
    '''
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards

    return discounted_rewards


def discount_and_normalize_rewards(all_rewards, discount_rate):
    '''
    Parameters:
    all_rewards: (num_game_rounds, game_steps). rewards for each iteration.

    Return:
    all_discounted_rewards: (num_game_rounds, game_steps). discounted rewards for each iteration.
    '''

    # calculate discounted rewards
    all_discounted_rewards = []
    for rewards in all_rewards:
        all_discounted_rewards.append(helper_discount_rewards(rewards, discount_rate))

    # normalize discounted rewards
    flat_rewards = np.concatenate(all_discounted_rewards)  # (num_game_rounds, game_steps)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]
 

if __name__ == '__main__':

    num_game_rounds = 10
    max_game_steps = 1000
    num_iterations = 300
    discount_rate = 0.9

    ########## Create Model ##########
    model_x, model_y, model_action, model_gradients, model_loss, model_gradient_placeholders, model_training_op = create_model()
    
    ########## Train Model ##########
    with tf.Session() as sess:
        initializer = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run(initializer)
        for iteration in range(num_iterations):
            print('On Iterations: {}'.format(iteration))

            all_rewards = []
            all_gradients = []
            all_x = []
            all_y = []
            
            for game in range(num_game_rounds):

                current_rewards = []
                current_gradients = []
                y = []

                observations = env.reset()
                score = 0
                for step in range(max_game_steps):
                    # env.render()
                    fetches = [model_action, model_gradients]
                    feed_dict = {model_x: observations.reshape((1, num_inputs))}
                    action, gradients = sess.run(fetches, feed_dict)

                    observations, reward, done, info = env.step(action[0][0])

                    current_rewards.append(reward)
                    current_gradients.append(gradients)
                    x.append(observations)
                    y.append(action[0][0])

                    score += reward
                    if done:
                        break
                print(score)
                all_rewards.append(current_rewards)
                all_gradients.append(current_gradients)
                all_x.append(x)
                all_y.append(y)

            all_discounted_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
            
            feed_dict = {model_y: all_y}
            for var_index, gradient_placeholder in enumerate(model_gradient_placeholders):
                # mean_gradients = np.sum([(np.sum(rewards) - 70) * all_gradients[game_index][step][var_index]
                #                         for game_index, rewards in enumerate(all_discounted_rewards)
                #                         for step, reward in enumerate(rewards)], axis=0) / num_game_rounds
                mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                      for step, reward in enumerate(rewards)], axis=0)
                # print(mean_gradients)
                feed_dict[gradient_placeholder] = mean_gradients

            sess.run(model_training_op, feed_dict=feed_dict)
        saver.save(sess, 'models/PG/model')
        
    
    avg_score = 0
    for i in range(20):
        observations = env.reset()
        with tf.Session() as sess:
            new_saver = tf.train.Saver()
            new_saver.restore(sess, 'models/PG/model')
            score = 0
            for x in range(500):
                env.render()

                fetches = [model_action]
                feed_dict = {model_x: observations.reshape((1, num_inputs))}
                action_val = sess.run(fetches, feed_dict)
                # print(action_val)
                observation, reward, done, info = env.step(action_val[0][0])
                score += reward

                if done:
                    print('Your Score Is: {}'.format(score))
                    avg_score += score
                    break
    print('Average Score: {}'.format(avg_score / 20))
