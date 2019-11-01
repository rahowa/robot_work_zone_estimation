from collections import deque
import numpy as np 
from numpy.linalg import inv 
import matplotlib.pyplot as plt 
from dataclasses import dataclass
from typing import List


@dataclass
class ErrorsEstimation:
    x_coord: float 
    y_coord: float 
    v_x:     float 
    v_y:     float  


@dataclass
class ErrorsMeasurement:
    x_coord: float 
    y_coord: float 
    v_x:     float 
    v_y:     float 
    

class KalmanFilter(object):
    def __init__(self, meas_errors: ErrorsMeasurement, est_errors: ErrorsEstimation):
        self.meas_errors = meas_errors
        self.est_errors = est_errors
        self.delta_t = 1.0
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
        self.num_of_observations = 4
        self.covariance_est = self.covariance(est_errors.x_coord,
                                              est_errors.y_coord,
                                              est_errors.v_x,
                                              est_errors.v_y)
        self.history = deque(maxlen=100)

    def update_speed(self, history: List[float]):
        if len(history) > 1:
            history = np.array(self.history)
            speed = (history[1:, :] - history[:-1, :]).mean(0)
            self.speed_x = speed[0]
            self.speed_y = speed[1]

    def update_acceleration(self, history: List[float]):
        if len(history) > 1:
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

    def predict(self, x: int, y: int, v_x: int,
                v_y: int, a:int) -> np.ndarray:
        X = np.array([
             [x],
             [y],
             [v_x],
             [v_y]
        ])
        result = self.transition_matrix_A.dot(X)
        result += self.transition_matrix_B.dot(a)
        return result

    def update(self, new_x: int, new_y: int) -> np.ndarray:
        self.history.append([new_x, new_y])
        self.update_speed(self.history)
        self.update_acceleration(self.history)
        X = self.predict(new_x, new_y, self.speed_x, self.speed_y, self.acceleration)
        self.covariance_est = np.diag(
            np.diag(
                self.transition_matrix_A.dot(self.covariance_est).dot(self.transition_matrix_A.T)
                )
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
        X = X + K.dot(Y - H.dot(X))
        # Update Process Covariance Matrix
        self.covariance_est = (np.identity(len(K)) - K.dot(H)).dot(self.covariance_est)
        return X

def main():
    x_observations = np.arange(0, 1000, 10) + np.random.normal(0, 1, 100)
    y_observations = np.sin(x_observations/10)+ np.random.normal(0, 1, 100)

    # Process / Estimation Errors
    # and
    # Observation Errors
    estimation_errors = ErrorsEstimation(20, 1, 5, 5)
    observation_errors = ErrorsMeasurement(25, 1, 6, 6)
    x_filtered = []
    y_filtered = []

    kf = KalmanFilter(observation_errors, estimation_errors)
    for timestep in range(x_observations.shape[0]):
        new_x = x_observations[timestep]
        new_y = y_observations[timestep]

        result = kf.update(new_x, new_y)
        print(result, result.shape)
        x_filtered.append(result[0][0])
        y_filtered.append(result[1][0])

    plt.plot(x_observations, y_observations, marker='o')
    plt.plot(x_filtered, y_filtered, marker='x')
    plt.show()





if __name__ == "__main__":
    main()

