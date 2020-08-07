__author__ = "Yuvaram singh"
__credits__ = ["Yuvaram singh", "rmgi"]
__email__ = "vmvijayayuvaram.s@hcl.com"


import numpy as np
import tensorflow as tf


class Replay_buffer:
    def __init__(
        self,
        env_requirment,
        data_points,
        observation_spec,
        action_to_int,
        custom_observation_spec=None,
    ):
        self.History = env_requirment["History"]
        self.observation_space = 0
        self.action_to_int = action_to_int
        if custom_observation_spec == None:
            for key, value in observation_spec.items():
                if value:
                    self.observation_space += data_points[key] * self.History
        else:
            self.observation_space = custom_observation_spec * self.History
        assert (
            self.observation_space > 0
        ), "The observation space should be greater than 0. check  env_requirment data_points observation_spec"

        self.buffer_size = env_requirment["replay_buffer"]
        self.count = 0
        self.state_memory = np.array(
            [[0.0] * self.observation_space] * self.buffer_size, dtype=np.float32
        )
        self.next_state_memory = np.array(
            [[0.0] * self.observation_space] * self.buffer_size, dtype=np.float32
        )
        self.action_memory = np.array([0] * self.buffer_size, dtype=np.int32)
        self.reward_memory = np.array([0] * self.buffer_size, dtype=np.float32)
        self.terminal_memory = np.array([0] * self.buffer_size, dtype=np.float32)

    def transition(self, state, action, reward, state_, terminate):
        index = self.count % self.buffer_size
        self.state_memory[index] = state
        self.action_memory[index] = self.action_to_int[action]
        self.reward_memory[index] = reward
        self.next_state_memory[index] = state_
        self.terminal_memory[index] = terminate
        self.count += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.count, self.buffer_size)
        batch_idx = np.random.choice(max_mem, batch_size, replace=False)

        state = self.state_memory[batch_idx]
        action = self.action_memory[batch_idx]
        reward = self.reward_memory[batch_idx]
        next_state = self.next_state_memory[batch_idx]
        terminal = self.terminal_memory[batch_idx]

        return state, action, reward, next_state, terminal


class RL_agent:
    def __init__(self, action_spec, Pre_trained=False):
        self.Pre_trained = Pre_trained
        self.action_spec = action_spec
        self.int_to_action = {0: "UP", 1: "DOWN", 2: "RIGHT", 3: "LEFT", 4: "BREAK"}
        self.action_to_int = {value: key for key, value in self.int_to_action.items()}
        self.setup_env_requirment()
        self.neural_network_init()
        self.episode = 0

    def neural_network_init(self):
        if not self.Pre_trained:
            # TODO: build a better Neural network
            self.Q_network = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        100, input_shape=(self.observation_space,), activation="relu"
                    ),
                    tf.keras.layers.Dense(100, activation="relu"),
                    tf.keras.layers.Dense(self.action_spec, activation=None),
                ]
            )

            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.env_requirment["learning_rate"]
            )

            self.Q_network.compile(optimizer=optimizer, loss="mean_squared_error")
        else:
            self.Q_network = tf.keras.models.load_model(
                self.env_requirment["model_save_path"]
            )

    def setup_env_requirment(self):
        # TODO modify the env_requirement to get better results. each one is a hyper parameter
        # modification of these will give you a better agent
        self.env_requirment = {
            "History": 5,
            "replay_buffer": 100,
            "learning_rate": 0.1,
            "max_no_of_episodes": 50,
            "update_no_of_episodes": 5,
            "discount_rate": 0.9,
            "epsilon": 0.3,
            "epsilon_decay": 1e-3,
            "epsilon_min": 0.1,
            "no_of_steps_to_train": 10,
            "batch_size": 100,
            "model_save_path": "./weight/model.h5",
        }
        self.data_points = {
            "lidar": 18,
            "position": 2,
            "angle": 1,
            "acceleration": 2,
            "velocity": 2,
            "steering": 1,
        }
        # TODO Try to use various combinations of observation space for observation_spec to get a better awareness about
        # the car and its current state.
        self.observation_spec = {
            "lidar": True,
            "position": False,
            "angle": False,
            "acceleration": False,
            "velocity": False,
            "steering": False,
        }
        self.replay_buffer = Replay_buffer(
            self.env_requirment,
            self.data_points,
            self.observation_spec,
            self.action_to_int,
        )
        self.observation_space = self.replay_buffer.observation_space
        self.epsilon = self.env_requirment["epsilon"]
        self.no_of_steps_to_train = self.env_requirment["no_of_steps_to_train"]

    # TODO build a better reward function utilizing various combinations of the provided informations
    def reward(
        self,
        lidar_end_pts,
        position,
        angle,
        acceleration,
        velocity,
        steering,
        crashed,
        previous_obs,
    ):
        if not crashed:
            return velocity[0]
        else:
            return -15

    def step(self, state, action, reward, state_, terminate):

        self.replay_buffer.transition(state, action, reward, state_, terminate)

    def train(self):
        if self.replay_buffer.count < self.env_requirment["batch_size"]:
            print(
                "Not training becaues of les no of samples ", self.replay_buffer.count
            )
            return
        for i in range(self.no_of_steps_to_train):
            (
                state,
                action,
                reward,
                next_state,
                terminal,
            ) = self.replay_buffer.sample_buffer(self.env_requirment["batch_size"])
            state = np.array(state)
            next_state = np.array(next_state)
            q_current = self.Q_network.predict(state)
            q_next_state = self.Q_network.predict(next_state)

            q_target = np.copy(q_current)
            batch_index = np.arange(self.env_requirment["batch_size"], dtype=np.int32)

            q_target[batch_index, action] = reward + self.env_requirment[
                "discount_rate"
            ] * np.max(q_next_state, axis=1) * (1 - terminal)
            self.Q_network.train_on_batch(state, q_target)
        self.epsilon = (
            self.epsilon - self.env_requirment["epsilon_decay"]
            if self.epsilon > self.env_requirment["epsilon_min"]
            else self.env_requirment["epsilon_min"]
        )

        self.Q_network.save(self.env_requirment["model_save_path"])

    # TODO change the observation parser if you have made changes to self.observation_spec
    def observation_parser(self, obs_dict):
        obs = obs_dict["Lidar"]
        return obs

    def take_action(self, observation):

        self.current_state = self.observation_parser(observation)
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_spec)
        else:

            obs = np.array([self.current_state])

            predictions = self.Q_network.predict(obs)
            action = np.argmax(predictions, axis=1)
            action = action[0]

        pas = action
        action = self.int_to_action[pas]

        return action
