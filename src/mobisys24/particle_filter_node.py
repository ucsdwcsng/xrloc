#!/usr/bin/env python3

# %% Imports
import rospy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Vector3Stamped
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import signal
from filterpy import monte_carlo as mc
from joblib import Parallel, delayed
from functools import partial


MAX_VEL = 0.03  # m/s
FRAME_RATE = 6  # Hz
MAX_PARTICLES = 2500
C_SPEED = 3e8
import time

# %% Setup env and create measurements
freq = 3.4944e9  # Hz
wavelen = 3e8 / freq
# 125 data
# tdoa_std = np.array([181.46, 179.47, 187.98, 306.95, 249.62,
#                      175.81, 169.52, 342.15, 220.47,
#                      179.40, 336.06, 216.88,
#                      346.88, 217.48,
#                      410.29]) * 1e-12 # in seconds
# pdoa_std = np.array([16.67, 15.25, 14.42, 25.85, 13.39,
#                      10.14, 7.63, 15.80, 8.74,
#                      11.03, 19.72, 10.29,
#                      19.35, 9.21,
#                      17.49]) * np.pi/180 # in radians
# 127 data

# beta = [1.44552772e-02,  4.37217877e-02,  4.25297196e-02,  3.83577072e-02,  4.21063355e-02,  6.69028286e-02]
# gamma = [1, 1, 1, 1, 1, 1]
# tdoa_std = 250e-12 # in seconds
# pdoa_std = 10 * np.pi/180 # in radians
# Load all data
# fname = "/home/aarun/Research/data/muloc/mULoc_csv/20230127_tdoa_pdoa_tracker_cali_exp20230127_v02/p7cal/" \
#         "20230127_move09walk_tdoa_pdoa_tracker_cali_exp20230127_v02.csv"
# fname = "/home/aarun/Research/data/muloc/mULoc_csv/20230127_tdoa_pdoa_tracker_cali_exp20230127_v02/p7cal/" \
#         "20230127_move04squarefast_tdoa_pdoa_tracker_cali_exp20230127_v02.csv"
# Post sigcomm, with different additive calibration applied
# fname = "/home/aarun/Research/data/muloc/mULoc_csv/20230223_tdoa_pdoa_tracker_optcal_exp20230127/bcal06p1p7p8p9std/" \
#         "20230301_move08butteflybig_tdoa_pdoa_tracker_bcal06p1p7p8p9std_exp20230127.csv"
# fname = "/home/aarun/Research/data/muloc/mULoc_csv/20230223_tdoa_pdoa_tracker_optcal_exp20230127/bcal06p1p7p8p9std/" \
#          "20230301_move01square_tdoa_pdoa_tracker_bcal06p1p7p8p9std_exp20230127.csv"



# %% Utility functions
def compute_tdoa(tx_loc, rx_loc):
   num_rx = rx_loc.shape[0]

   tdoa = np.zeros((num_rx, num_rx))
   for i in range(num_rx):
      for j in range(num_rx):
         if j > i:
            ti = np.linalg.norm(tx_loc - rx_loc[i]) / C_SPEED
            tj = np.linalg.norm(tx_loc - rx_loc[j]) / C_SPEED
            tdoa[i, j] = ti - tj

   return tdoa

def compute_tdoa_matrix(tx_loc, rx_loc):

   times = np.linalg.norm(tx_loc[:, None] - rx_loc[None], axis=2) / C_SPEED
   tdoa_matrix = times[..., None] - times[:, None]

   return tdoa_matrix


def compute_pdoa(tx_loc, rx_loc, wavelen, beta=None, gamma=None):
   num_rx = rx_loc.shape[0]
   pdoa = np.zeros((num_rx, num_rx))

   for i in range(num_rx):
      for j in range(num_rx):
         if j > i:
            pi = 2 * np.pi / wavelen * (np.linalg.norm(tx_loc - rx_loc[i]) % wavelen)
            pj = 2 * np.pi / wavelen * (np.linalg.norm(tx_loc - rx_loc[j]) % wavelen)

            if beta is not None and gamma is not None:
               di = np.linalg.norm(tx_loc - rx_loc[i])*1e3
               pi -= (beta[i] * (di ** gamma[i]))*np.pi/180
               dj = np.linalg.norm(tx_loc - rx_loc[j])*1e3
               pj -= (beta[j] * (dj ** gamma[j]))*np.pi/180

            pdoa[i, j] = np.angle(np.exp(1j * (pi - pj)))

   return pdoa

def compute_pdoa_matrix(tx_loc, rx_loc, wavelen, beta=None, gamma=None):

   distances = np.linalg.norm(tx_loc[:, None] - rx_loc[None], axis=2)
   phases = 2 * np.pi / wavelen * distances
   if beta is not None:
      phases -= (beta[None] * ((distances * 1e3) ** gamma[None])) * np.pi / 180
   pdoa_matrix = np.angle(np.exp(1j * (phases[..., None] - phases[:, None])))
   return pdoa_matrix


def multivariate_normal_pdf(x, mean, covariance):
   k = len(mean)
   norm_const = 1.0 / (np.sqrt((2 * np.pi) ** k * np.linalg.det(covariance)))
   x_mu = np.matrix(x - mean)
   inv_covmat = np.linalg.inv(covariance)
   inner_exp = np.exp(-0.5 * np.dot(np.dot(x_mu, inv_covmat), x_mu.T))
   return norm_const * np.squeeze(np.asarray(inner_exp))

def multivariate_normal_pdf_matrix(x, mean, covariance):
   k = mean.shape[1]
   norm_const = 1.0 / (np.sqrt((2 * np.pi) ** k * np.linalg.det(covariance[0])))
   x_mu = np.matrix(x - mean)
   inv_covmat = np.linalg.inv(covariance)
   inv_covmat = np.tile(inv_covmat, (len(x_mu), 1, 1))
   inner_exp = np.exp(-0.5 * ((x_mu[:, None] @ inv_covmat)[:, None] @ x_mu[..., None])).T
   return norm_const * np.squeeze(np.asarray(inner_exp))



def compute_joint_prob(tx_loc, rx_loc, tdoa_meas, tdoa_var,
                       pdoa_meas, pdoa_var, wavelen, beta, gamma):
   """
   Computes the total probability of the tx_loc given the Pdoa and Tdoa measurments along with their noise variances.
   This function internally converts time to ns to bring time and phase units to same scale

   tx_loc = [x, y] locations of the transmitter
   rx_loc = array of receiver locations, [num_rx x 2]
   tdoa_meas = pairwise TDoa between two receivers t_j - t_i, j > i, upper triangular matrix with zero diagonals,
                  [num_rx x num_rx]
   tdoa_var = in seconds, the variance of tdoa measurements
   pdoa_meas =  pairwise PDoA between two receivers t_j - t_i, j > i, upper triangular matrix with zero diagonals,
                     [num_rx x num_rx]
   pdoa_var = in seconds, the variance of tdoa measurements
   wavelen = wavelength of carrier wave
   beta = scaling bias wrt distance
   gamma = exponential bias wrt distance
   """

   num_rx = rx_loc.shape[0]
   tx_loc = np.array(tx_loc)

   pdoa_exp = compute_pdoa(tx_loc, rx_loc, wavelen, beta, gamma)
   measurement_error = np.angle(np.exp(1j*(pdoa_meas - pdoa_exp)))
   error_vec_pdoa = list(measurement_error[np.triu_indices(num_rx, k=1)])

   tdoa_exp = compute_tdoa(tx_loc, rx_loc)
   measurement_error = (tdoa_meas - tdoa_exp) * 1e9  # convert tdoa in ns
   error_vec_tdoa = list(measurement_error[np.triu_indices(num_rx, k=1)])

   combined_error = np.array(error_vec_tdoa + error_vec_pdoa)

   if type(pdoa_var) is np.ndarray:
      sigma = np.diag(list(tdoa_var * 1e18) + list(pdoa_var))
   else:
      sigma = np.diag([tdoa_var * 1e18] * len(error_vec_tdoa) +
                      [pdoa_var] * len(error_vec_pdoa))


   mean = np.array([0] * (2 * len(error_vec_tdoa)))
   prob = multivariate_normal_pdf(combined_error, mean, sigma)

   return prob

def compute_joint_prob_matrix(tx_loc, rx_loc, tdoa_meas, tdoa_var,
                       pdoa_meas, pdoa_var, wavelen, beta, gamma):
   """
   Computes the total probability of the tx_loc given the Pdoa and Tdoa measurments along with their noise variances.
   This function internally converts time to ns to bring time and phase units to same scale

   tx_loc = n x [x, y] locations of the transmitter
   rx_loc = array of receiver locations, [num_rx x 2]
   tdoa_meas = pairwise TDoa between two receivers t_j - t_i, j > i, upper triangular matrix with zero diagonals,
                  [num_rx x num_rx]
   tdoa_var = in seconds, the variance of tdoa measurements
   pdoa_meas =  pairwise PDoA between two receivers t_j - t_i, j > i, upper triangular matrix with zero diagonals,
                     [num_rx x num_rx]
   pdoa_var = in seconds, the variance of tdoa measurements
   wavelen = wavelength of carrier wave
   beta = scaling bias wrt distance
   gamma = exponential bias wrt distance
   """

   num_rx = rx_loc.shape[0]
   tx_loc = np.array(tx_loc)
   num_locs = len(tx_loc)
   indices = np.triu_indices(num_rx, k=1)

   pdoa_exp = compute_pdoa_matrix(tx_loc, rx_loc, wavelen, beta, gamma)
   measurement_error = np.angle(np.exp(1j*(pdoa_meas[None] - pdoa_exp)))
   error_vec_pdoa = measurement_error[:, indices[0], indices[1]]

   tdoa_exp = compute_tdoa_matrix(tx_loc, rx_loc)
   measurement_error = (tdoa_meas[None] - tdoa_exp) * 1e9  # convert tdoa in ns
   error_vec_tdoa = measurement_error[:, indices[0], indices[1]]

   combined_error = np.hstack((error_vec_tdoa, error_vec_pdoa))


   if type(pdoa_var) is np.ndarray:
      sigma = np.diag(list(tdoa_var * 1e18) + list(pdoa_var))
   else:
      sigma = np.diag([tdoa_var * 1e18] * error_vec_tdoa.shape[1] +
                      [pdoa_var] * error_vec_pdoa.shape[1])


   mean = np.zeros(combined_error.shape)
   prob = multivariate_normal_pdf_matrix(combined_error, mean, sigma[None])

   return prob

def initialize_particles(lower_left, upper_right, num_particles=MAX_PARTICLES):
   """
   :param grid_len: size of grid in meters, assumed to be square grid
   :param num_particles: total number of particles intialized
   :return: locations of particles, numpy array [MAX_PARTICLES, 2]
   """
   loc_x = stats.uniform.rvs(loc=lower_left[0], scale=upper_right[0], size=num_particles)
   loc_y = stats.uniform.rvs(loc=lower_left[1], scale=upper_right[1], size=num_particles)
   return np.vstack((loc_x, loc_y)).T


def move_particles(particles, max_velocity=MAX_VEL, time_step=1 / FRAME_RATE):
   """
   :param particles: locations of all particles at current time step t numpy array, [MAX_PARTICLES x 2]
   :param max_velocity: presumed max location of movement of hand
   :param time_step: time difference between two subsequent discrete measurements
   :return: locations of all particles at next timestep t+1
   """
   # mv_norm = stats.multivariate_normal(np.array([0, 0]),
   #                                     np.array([[np.abs(max_velocity[0]) / 3 * time_step, 0],
   #                                               [0, np.abs(max_velocity[1]) / 3 * time_step]]),
   #                                     allow_singular=True)
   # movements = mv_norm.rvs(MAX_PARTICLES)
   particles[:, 0] += np.random.normal(0, np.sqrt(np.abs(max_velocity[0]) / 3 * time_step), len(particles))
   particles[:, 1] += np.random.normal(0, np.sqrt(np.abs(max_velocity[1]) / 3 * time_step), len(particles))
   return particles


def update_subproc(iter, particles, rx_loc, tdoa_meas, pdoa_meas, tdoa_var, pdoa_var, wavelen, beta, gamma):
   val = compute_joint_prob(particles[iter], rx_loc, tdoa_meas, tdoa_var, pdoa_meas, pdoa_var, wavelen, beta, gamma)
   return val


def update_particles(particles, rx_loc, tdoa_meas, pdoa_meas, tdoa_var, pdoa_var, wavelen, beta, gamma):
   """
   :param particles: locations of all particles at current time step t numpy array, [MAX_PARTICLES x 2]
   :param values: probability score of above locations
   :param rx_loc: receiver locations, [NUM_RX x 2]
   :param tdoa_meas: tdoa measurements made across antenna pairs, in sec [NUM_RX x NUM_RX]
   :param pdoa_meas: pdoa measurements made across antenna pairs, in radians [NUM_RX x NUM_RX]
   :param tdoa_var: tdoa measurement variance, in sec
   :param pdoa_var: pdoa measurement variance, in radians
   :param wavelen: center frequency UWB wavelength
   :return: updates probability values given the measurements
   """
   # cur_func = partial(update_subproc, particles=particles, rx_loc=rx_loc, tdoa_meas=tdoa_meas, pdoa_meas=pdoa_meas,
   #                    tdoa_var=tdoa_var, pdoa_var=pdoa_var, wavelen=wavelen, beta=beta, gamma=gamma)
   # values = Parallel(n_jobs=10)(delayed(cur_func)(jj) for jj in range(len(particles)))
   # values = np.array(values)[:, 0, 0]
   # values = values / np.linalg.norm(particles - np.median(particles, axis=0), axis=1)**0.8

   # For debugging purposes
   # values = np.zeros((len(particles)))
   # for ii, p in enumerate(particles):
   #    values[ii] = compute_joint_prob(p, rx_loc, tdoa_meas, tdoa_var, pdoa_meas, pdoa_var, wavelen, beta, gamma)

   values = compute_joint_prob_matrix(particles, rx_loc, tdoa_meas, tdoa_var, pdoa_meas, pdoa_var, wavelen, beta, gamma)

   # normalize the values
   if np.sum(values) != 0:
      values /= np.sum(values)
   else:
      values = np.ones((len(particles))) / len(particles)
   return values

def update_particles_matrix(particles, rx_loc, tdoa_meas, pdoa_meas, tdoa_var, pdoa_var, wavelen, beta, gamma):
   """
   :param particles: locations of all particles at current time step t numpy array, [MAX_PARTICLES x 2]
   :param values: probability score of above locations
   :param rx_loc: receiver locations, [NUM_RX x 2]
   :param tdoa_meas: tdoa measurements made across antenna pairs, in sec [NUM_RX x NUM_RX]
   :param pdoa_meas: pdoa measurements made across antenna pairs, in radians [NUM_RX x NUM_RX]
   :param tdoa_var: tdoa measurement variance, in sec
   :param pdoa_var: pdoa measurement variance, in radians
   :param wavelen: center frequency UWB wavelength
   :return: updates probability values given the measurements
   """
   values = compute_joint_prob_matrix(particles, rx_loc, tdoa_meas, tdoa_var, pdoa_meas, pdoa_var, wavelen, beta, gamma)

   # normalize the values
   if np.sum(values) != 0:
      values /= np.sum(values)
   else:
      values = np.ones((len(particles))) / len(particles)
   return values



class MeasurementProcessor:
   def __init__(self):
      rospy.init_node('measurement_processor', anonymous=True)
      self.data_sub = rospy.Subscriber("combined_data", Float64MultiArray, self.update_data)
      self.result_pub = rospy.Publisher("result_data", Vector3Stamped, queue_size=10)
      self.particles = None
      self.tdoa_std = 2*np.array([97.80, 131.37, 107.13, 108.73, 138.82,
                     127.44, 86.75, 96.32, 122.84,
                     134.04, 128.98, 150.57,
                     105.26, 113.34,
                     118.40]) * 1e-12
      self.pdoa_std = 4*np.array([2.09, 1.97, 2.40, 2.17, 2.15,
                     2.27, 2.35, 2.29, 1.86,
                     2.71, 2.17, 2.55,
                     2.47, 2.39,
                     2.21]) * np.pi/180
      self.beta = [5.58707221e-01, 1.61452039e-01, 9.15229728e-01, 9.99766325e-01, 9.15852658e-02, 5.82070621e-01]
      self.gamma = [7.75880355e-01, 9.47910280e-01, 7.38824199e-01, 7.47507817e-01, 9.99924725e-01, 8.18594362e-01]
      #self.beta = [1.44552772e-02,  4.37217877e-02,  4.25297196e-02,  3.83577072e-02,  4.21063355e-02,  6.69028286e-02]
      #self.gamma = [1, 1, 1, 1, 1, 1]

   def update_data(self, data):
      combined_flat = np.array(data.data)
      num_elements = len(combined_flat) // 2  # TDoAとPDoAは同じ数の要素を持つ
      tdoa_flat = combined_flat[:num_elements]
      pdoa_flat = combined_flat[num_elements:]
      
      tdoa = self.reconstruct_matrix(tdoa_flat)
      pdoa = self.reconstruct_matrix(pdoa_flat)
      self.localization(tdoa, pdoa)

   def publish_result(self, position):
      # 計算された位置をパブリッシュ
      msg = Vector3Stamped()
      msg.header.stamp = rospy.Time.now()
      msg.vector.x = position[0]
      msg.vector.y = position[1]
      msg.vector.z = 0  # 3D位置が必要な場合はここを調整
      self.result_pub.publish(msg)
      rospy.loginfo("Publishing result data")

   def reconstruct_matrix(self, flat_data):
      # 与えられた1次元配列から行列の次元を計算
      num_rx_max = 6
      mat = np.zeros((num_rx_max, num_rx_max))

      # np.triu_indicesを使って上三角部分のインデックスを取得
      indices = np.triu_indices(num_rx_max, k=1)

      # 上三角部分にflat_dataを割り当て
      mat[indices] = flat_data

      return mat

   def localization(self, err_tdoa, err_pdoa):
      # TODO: make gnd truth data
      # gnd_locations = []
      # offset = np.min(gnd_locations, axis=0)
      # gnd_locations -= offset
      err_tdoa = err_tdoa * 1e-12
      
      # print(err_tdoa)
      velocities = []
   
      # num_loc = len(gnd_locations)
      aps_to_remove = []
      num_rx_max = 6
   
      self.beta = np.delete(self.beta, aps_to_remove)
      self.gamma = np.delete(self.gamma, aps_to_remove)
   
      # define receiver locations
      rx_locations = np.array([[2.0, 5.0],
                              [2.2, 5.0],
                              [2.4, 5.0],
                              [2.6, 5.0],
                              [2.8, 5.0],
                              [3.0, 5.0]])
   
      # rx_locations -= offset
      rx_locations = np.delete(rx_locations, aps_to_remove, axis=0)
   
      num_rx = rx_locations.shape[0]
   
      indices = np.triu_indices(num_rx_max, k=1)
   
      # delete the extra stuff
      tdoa_std_mat = np.zeros((num_rx_max, num_rx_max))
      tdoa_std_mat[indices[0], indices[1]] = self.tdoa_std
      tdoa_std_mat = np.delete(np.delete(tdoa_std_mat, aps_to_remove, axis=0), aps_to_remove, axis=1)
      # 除外すべきデータを消している(ここでは動作していない)
   
      pdoa_std_mat = np.zeros((num_rx_max, num_rx_max))
      pdoa_std_mat[indices[0], indices[1]] = self.pdoa_std
      pdoa_std_mat = np.delete(np.delete(pdoa_std_mat, aps_to_remove, axis=0), aps_to_remove, axis=1)
      # 上記と同様
   
      err_tdoa = np.delete(np.delete(err_tdoa, aps_to_remove, axis=0), aps_to_remove, axis=1)
      err_pdoa = np.delete(np.delete(err_pdoa, aps_to_remove, axis=0), aps_to_remove, axis=1)
      # 上記と同様
   
      indices = np.triu_indices(num_rx, k=1)
      
      self.tdoa_std = tdoa_std_mat[indices[0], indices[1]]
      self.pdoa_std = pdoa_std_mat[indices[0], indices[1]]
   
      # apply median filtering
      # err_tdoa = signal.medfilt(err_tdoa, kernel_size=[5, 1, 1])
      # err_pdoa = signal.medfilt(err_pdoa, kernel_size=[5, 1, 1])
   
      # # Compute gnd truth pdoa and tdoa
      # gnd_tdoa = []
      # for loc in gnd_locations:
      #    gnd_tdoa.append(compute_tdoa(loc, rx_locations))
      # gnd_tdoa = np.array(gnd_tdoa)
   
      # gnd_pdoa = []
      # gnd_pdoa_bias = []
      # for loc in gnd_locations:
      #    gnd_pdoa.append(compute_pdoa(loc, rx_locations, wavelen))
      #    gnd_pdoa_bias.append(compute_pdoa(loc, rx_locations, wavelen, beta, gamma))
      # gnd_pdoa = np.array(gnd_pdoa)
      # gnd_pdoa_bias = np.array(gnd_pdoa_bias)
   
      # %% Run particle filter on measurements
      center = [3.5, 3.5]
      # center = gnd_locations[0]
      sq_len = 3
      if self.particles is None:
         self.particles = initialize_particles([0, 0], [sq_len, sq_len]) + center - (sq_len / 2, sq_len / 2)
   
      # particles = initialize_particles([0, 0], [sq_len, sq_len]) + center - (sq_len / 2, sq_len / 2)
   
      values = np.ones((MAX_PARTICLES,)) / MAX_PARTICLES
      all_errors = []
      # all_predictions = np.zeros((num_loc, 2))
      compute_times = []
      should_plot = False
   
      if should_plot:
         fig = plt.figure()
         fig.add_subplot(111)
         plt.scatter(rx_locations[:, 0], rx_locations[:, 1], marker="^", label="RX Locations")
         plt.scatter(self.particles[:, 0], self.particles[:, 1], label="Particle Locations")
         plt.scatter(gnd_locations[0, 0], gnd_locations[0, 1], label="Gnd Location")  # this will change for moving points
         plt.legend()
         plt.gca().set_aspect("equal")
   
      prev_pred = None
      pred_loc = None
      pred_jump = 0
   
      tic = time.time()
      prev_pred = pred_loc
   
      self.particles = move_particles(self.particles, max_velocity=[0.02,0.02])
   
      values_matrix = update_particles_matrix(self.particles, rx_locations, err_tdoa, err_pdoa,
                              self.tdoa_std ** 2, self.pdoa_std ** 2, wavelen, self.beta, self.gamma)
   
      new_indices = mc.stratified_resample(values_matrix)
      print(values_matrix)
   
      # if ii == num_loc // 5:
      #    uniq, counts = np.unique(new_indices, return_counts=True)
      #    new_indices = np.repeat(uniq, np.ceil(counts / 10).astype(int))
   
      self.particles = self.particles[new_indices]
      pred_loc = np.mean(self.particles, axis=0)
   
      # all_predictions[ii, :] = pred_loc
      compute_times.append(time.time() - tic)
   
      # all_errors.append(np.linalg.norm(pred_loc - gnd_locations[ii]))
   
      # print(f"{ds}: Median compute time: {np.median(compute_times)}")
      # %% Plot all the error over time
      # all_errors = np.array(all_errors)
      print(pred_loc)

      
      self.publish_result(pred_loc)
    
def main():
    processor = MeasurementProcessor()
    rospy.spin()

if __name__ == '__main__':
    main()
