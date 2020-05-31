from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np 
from nptyping import Array
from numpy.linalg import inv 
import matplotlib.pyplot as plt 


@dataclass
class Errors:
    x_coord: float 
    y_coord: float 
    v_x:     float 
    v_y:     float  


class KalmanFilter(object):
    def __init__(self, meas_errors: Errors, est_errors: Errors, 
                 use_acceleration: bool):
        self.meas_errors = meas_errors
        self.est_errors = est_errors
        self.delta_t = 1.0
        self.use_acceleration = use_acceleration
        self.transition_matrix_A = np.array([
            [1 ,0, self.delta_t, 0],
            [0, 1, 0, self.delta_t],
            [0, 0, 1, 0           ],
            [0, 0, 0, 1           ]
                                        ])
        self.transition_matrix_B = np.array([
            [0.5 * (self.delta_t**2)],
            [0.5 * (self.delta_t**2)],
            [self.delta_t           ],
            [self.delta_t           ]
                                        ])
        self.speed_x = 0 
        self.speed_y = 0
        self.acceleration = 0
        self.min_n_of_samples = 4
        self.covariance_est = self.covariance(est_errors.x_coord,
                                              est_errors.y_coord,
                                              est_errors.v_x,
                                              est_errors.v_y)
        self.history: List[Tuple[float, float]] = list()
        self.prev_state = np.zeros((4, 1))

    def update_speed(self, history: List[Tuple[float, float]]):
        if len(history) > self.min_n_of_samples:
            history = np.array(self.history)
            speed = (history[1:, :] - history[:-1, :]).mean(0)
            self.speed_x = speed[0]
            self.speed_y = speed[1]

    def update_acceleration(self, history: List[Tuple[float, float]]):
        if len(history) > self.min_n_of_samples:
            history = np.array(self.history)
            speed = (history[1:, :] - history[:-1, :])
            self.acceleration = (speed[1:, :] - speed[:-1, :]).mean()

    def covariance(self, sigma1: float, sigma2: float,
                  sigma3: float, sigma4: float) -> np.ndarray:
        cov_matrix = np.array([
            [sigma1**2,       sigma2 * sigma1, sigma3 * sigma1, sigma4 * sigma1],
            [sigma1 * sigma2, sigma2**2,       sigma3 * sigma2, sigma4 * sigma2],
            [sigma1 * sigma3, sigma2 * sigma3, sigma3**2,       sigma4 * sigma3],
            [sigma1 * sigma4, sigma2 * sigma4, sigma3 * sigma4, sigma4**2      ],
        ])
        return np.diag(np.diag(cov_matrix))

    def predict(self, x: float, y: float, v_x: float, v_y: float) -> Array[float]:
        if len(self.history) >= self.min_n_of_samples:
            X = np.array([
                [x],
                [y],
                [v_x],
                [v_y]
            ])
            result = self.transition_matrix_A.dot(X)
            result += self.transition_matrix_B.dot(self.acceleration)
        else:
            result = np.array([[x], [y], [v_x], [v_y]])
        return result

    def update(self, new_x: float, new_y: float, X_: Array[float]) -> Array[float]:
        self.covariance_est = np.diag(
            np.diag(self.transition_matrix_A.dot(self.covariance_est).dot(self.transition_matrix_A.T))
            )
        # Calculating the Kalman Gain
        H = np.identity(self.transition_matrix_A.shape[1])
        R = self.covariance(self.meas_errors.x_coord, self.meas_errors.y_coord,
                            self.meas_errors.v_x, self.meas_errors.v_y)
        S = H.dot(self.covariance_est).dot(H.T) + R
        K = self.covariance_est.dot(H).dot(inv(S))
        # Reshape the new data into the measurement space.
        data = np.array([
            [new_x],
            [new_y],
            [self.speed_x],
            [self.speed_y]
        ])
        Y = H.dot(data).reshape(self.transition_matrix_A.shape[1], -1)
        # Update the State Matrix
        # Combination of the predicted state, measured values, covariance matrix and Kalman Gain
        X_ = X_ + K.dot(Y - H.dot(X_))
        # Update Process Covariance Matrix
        self.covariance_est = (np.identity(len(K)) - K.dot(H)).dot(self.covariance_est)
        return X_

    def step(self, new_x: float, new_y: float):
        self.history.append((new_x, new_y))
        self.update_speed(self.history)
        X_pred = self.predict(self.prev_state[0][0], self.prev_state[1][0], 
                              self.prev_state[2][0], self.prev_state[3][0])

        if self.use_acceleration:
            self.update_acceleration(self.history)

        if len(self.history) < self.min_n_of_samples:
            X_current =  np.array([[new_x], [new_y], [0], [0]])
        else:
            X_current = self.update(new_x, new_y, X_pred)

        self.prev_state = X_current
        return X_current


def main():
    x_observations = np.arange(0, 1000, 10) + np.random.normal(0, 1, 100)
    y_observations = np.sin(np.arange(0, 1000, 10)/10)+ np.random.normal(0, 1, 100)
    estimation_errors = Errors(10, 2, 1, 1)
    observation_errors = Errors(10, 1, 1, 1)
    x_filtered = [x_observations[0]]
    y_filtered = [y_observations[0]]

    kf = KalmanFilter(observation_errors, estimation_errors, False)
    for timestamp in range(1, x_observations.shape[0]-1):
        new_x = x_observations[timestamp]
        new_y = y_observations[timestamp]
        result = kf.step(new_x, new_y)
        x_filtered.append(result[0][0])
        y_filtered.append(result[1][0])
        print(result, result.shape, kf.acceleration)

    plt.style.use("ggplot")
    plt.plot(np.arange(0, 1000, 10), np.sin(np.arange(0, 1000, 10)/10),
             '--', label='original signal')
    plt.plot(x_observations, y_observations, marker='o', 
            label='noised signal')
    plt.plot(x_filtered, y_filtered, marker='x', label='filtered signal')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    main()

