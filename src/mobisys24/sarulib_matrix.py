import time
import json
import math
import cmath
import numpy as np
import scipy.constants
from joblib import Parallel, delayed
from scipy import stats


FREQ = 3.4944
LAMBDA = 1 / (FREQ * (10 ** 9)) * scipy.constants.c * (10 ** 3)
SIDELENGTH = 3000
n_jobs = -1



def angle2complex_array(angle_array):
    M, N = angle_array.shape
    
    ret = np.array([0j] * (M * N))
    ret = ret.reshape(M, N)

    for i in range(M):
        for j in range(N):
            ret[i][j] = angle2complex(angle_array[i][j])

    return ret
    

def get_fpi_phase(filename):
    filename = filename.replace("json", "cir")
    samples = get_cir_samples(filename)
    fpi = samples[10]
    return fpi


def get_cir_samples(filename):
    fp = open(filename, "r")
    line = fp.readline()
    line = fp.readline()

    ret = []
    while line:
        items = line.split(',')
        index = int(items[0])
        I = float(items[1])
        Q = float(items[2])

        ret.append(complex(I, Q))

        line = fp.readline()

    return ret
 
 

def get_json_data(filename):
    fp = open(filename, "r")
    items = json.load(fp)
    return items["seq"], items["FP_INDEX"], items["RCPHASE"], items["RX_STAMP"], items["RXTOFS"], items['DRX_CAR_INT'], items["RX_RAWST"], items["LDE_PPINDX"], items["RXPACC"], items["RXPACC_NOSAT"], items["CIR_PWR"], items["FP_AMPL1"], items["FP_AMPL2"], items["FP_AMPL3"], items["TC_SARL"]


def get_json_data_fast(filename):
    fp = open(filename, "r")
    items = json.load(fp)
    return items["seq"], items["RCPHASE"], items["RX_STAMP"], items["FPI_I"], items["FPI_Q"]


def time2tdoa(time_array):
   N = len(time_array)
   ret = np.zeros((N, N))
   for i in range(N):
      for j in range(N):
         ret[i, j] = time_array[i] - time_array[j]

   return ret


def phase2pdoa(phase_array):
   N = len(phase_array)
   ret = np.array([0j] * (N * N))
   ret = ret.reshape(N, N)

   for i in range(N):
      for j in range(N):
         ret[i, j] = phase_array[i] / phase_array[j]

   return ret



def calc_tdoa(tx_loc, rx_loc):
   num_rx = rx_loc.shape[0]

   tdoa = np.zeros((num_rx, num_rx))
   for i in range(num_rx):
      for j in range(num_rx):
         if j > i:
            ti = np.linalg.norm(tx_loc - rx_loc[i]) / scipy.constants.c
            tj = np.linalg.norm(tx_loc - rx_loc[j]) / scipy.constants.c
            tdoa[i, j] = tj - ti
            tdoa[j, i] = - tdoa[i, j]

   return tdoa



def angle2complex(angle):
    theta = math.pi * angle / 180
    return complex(math.cos(theta), math.sin(theta))


def calc_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calc_distance_3d(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)



def calc_correlation(a, b):
#    a = copy.copy(a)
#    b = copy.copy(b)
#    a = (a - np.mean(a)) / (np.std(a) * len(a))
#    b = (b - np.mean(b)) / (np.std(b))
#    print(a)
#    print(b)

    val = np.correlate(a, b)
    return cmath.phase(val)
#    return abs(val)
    


def get_phase_array(loc_tag, loc_anchor, error_sigma):
#    x_tag = x_tag * 10
#    y_tag = y_tag * 10

   x_tag = loc_tag[0]
   y_tag = loc_tag[1]

   size = len(loc_anchor)

   ret = np.array([0j] * size)

   index = 0
   for i in range(size):
      phase_error = np.random.normal(0.0, error_sigma)

      x_anchor = loc_anchor[i, 0]
      y_anchor = loc_anchor[i, 1]
#      x_anchor = loc_anchor[i][0]
#      y_anchor = loc_anchor[i][1]
      d = calc_distance(x_tag, y_tag, x_anchor, y_anchor)
      diff = d % LAMBDA
      angle = 360 * (diff / LAMBDA) + phase_error
      a0 = angle2complex(angle)
      ret[i] = a0

   return ret

def get_phase_array_expected_matrix(loc_tag, loc_anchor):
#    x_tag = x_tag * 10
#    y_tag = y_tag * 10

   distances = np.linalg.norm(tx_loc[:, None] - rx_loc[None], axis=2)
   phases = 2 * np.pi / LAMBDA * distances

   pdoa_matrix = np.angle(np.exp(1j * (phases[..., None] - phases[:, None])))
   
   return ret



def get_phase_array_3d(loc_tag, loc_anchor, error_sigma):
#    x_tag = x_tag * 10
#    y_tag = y_tag * 10

   x_tag = loc_tag[0]
   y_tag = loc_tag[1]
   z_tag = loc_tag[2]

   size = len(loc_anchor)

   ret = np.array([0j] * size)

   index = 0
   for i in range(size):
      phase_error = np.random.normal(0.0, error_sigma)

#      x_anchor = loc_anchor[i, 0]
#      y_anchor = loc_anchor[i, 1]
      x_anchor = loc_anchor[i][0]
      y_anchor = loc_anchor[i][1]
      z_anchor = loc_anchor[i][2]
      d = calc_distance_3d(x_tag, y_tag, z_tag, x_anchor, y_anchor, z_anchor)
      diff = d % LAMBDA
      angle = 360 * (diff / LAMBDA) + phase_error
      a0 = angle2complex(angle)
      ret[i] = a0

   return ret


def get_time_array_matrix(loc_tag, loc_anchor, error_sigma):
   x_tag = loc_tag[0]
   y_tag = loc_tag[1]

   size = len(loc_anchor)
   ret = np.array([0.0] * size)

   index = 0
   for i in range(size):
      time_error = np.random.normal(0.0, error_sigma)
      x_anchor = loc_anchor[i, 0]
      y_anchor = loc_anchor[i, 1]
      d = calc_distance(x_tag, y_tag, x_anchor, y_anchor)
      t = (d / scipy.constants.c) * 1e9 + time_error
      ret[i] = t

   return ret



def get_time_array(loc_tag, loc_anchor, error_sigma):
   x_tag = loc_tag[0]
   y_tag = loc_tag[1]

   size = len(loc_anchor)
   ret = np.array([0.0] * size)

   index = 0
   for i in range(size):
      time_error = np.random.normal(0.0, error_sigma)
      x_anchor = loc_anchor[i, 0]
      y_anchor = loc_anchor[i, 1]
      d = calc_distance(x_tag, y_tag, x_anchor, y_anchor)
      t = (d / scipy.constants.c) * 1e9 + time_error
      ret[i] = t

   return ret



def get_time_array_3d(loc_tag, loc_anchor, error_sigma):
   x_tag = loc_tag[0]
   y_tag = loc_tag[1]
   z_tag = loc_tag[2]

   size = len(loc_anchor)
   ret = np.array([0.0] * size)

   index = 0
   for i in range(size):
      time_error = np.random.normal(0.0, error_sigma)
      x_anchor = loc_anchor[i, 0]
      y_anchor = loc_anchor[i, 1]
      z_anchor = loc_anchor[i, 2]
      d = calc_distance_3d(x_tag, y_tag, z_tag, x_anchor, y_anchor, z_anchor)
      t = (d / scipy.constants.c) * 1e9 + time_error
      ret[i] = t

   return ret



def complex2angle(val):
    return np.angle(val, True)

def complex2angle_array(src):
#    print(src.shape)
    M, N = src.shape
    ret = np.array([0.0] * (N * M))
    ret = ret.reshape(M, N)
    
    for i in range(M):
        for j in range(N):
            ret[i][j] = complex2angle(src[i][j])

    return ret


def calc_likelihood_joint_matrix(loc_tag, loc_anchor, tdoa_measured,
                          pdoa_measured, tdoa_sigma, pdoa_sigma):
   N = loc_anchor.shape[0]
   M = (N  * (N - 1)) // 2

#   phase_array = get_phase_array(loc_tag, loc_anchor, 0)
   phase_array = get_phase_array_expected_matrix(loc_tag, loc_anchor)
   pdoa_ref = phase2pdoa(phase_array)
   
   error_measured = pdoa_measured / pdoa_ref
   error_measured_angle = complex2angle_array(error_measured)
   error_vec_pdoa = error_measured_angle[np.triu_indices(N, k=1)]
   pdoa_var = pdoa_sigma * pdoa_sigma

   time_array = get_time_array(loc_tag, loc_anchor, 0)
   tdoa_ref = time2tdoa(time_array)   

   error_measured = tdoa_measured - tdoa_ref
   error_vec_tdoa = error_measured[np.triu_indices(N, k=1)]
   tdoa_var = tdoa_sigma * tdoa_sigma
   
   cov_matrix = np.diag([tdoa_var] * M + [pdoa_var] * M)
   
   ### Multivariable Normal Random Variable
   mnrv = stats.multivariate_normal([0] * (M * 2), cov_matrix,
                                    allow_singular=True)

   error_vec = np.concatenate([error_vec_tdoa, error_vec_pdoa], 0)
   l = mnrv.pdf(error_vec)

   return l




def calc_likelihood_joint(loc_tag, loc_anchor, tdoa_measured,
                          pdoa_measured, tdoa_sigma, pdoa_sigma):
   N = loc_anchor.shape[0]
   M = (N  * (N - 1)) // 2

   phase_array = get_phase_array(loc_tag, loc_anchor, 0)
   pdoa_ref = phase2pdoa(phase_array)
   
   error_measured = pdoa_measured / pdoa_ref
   error_measured_angle = complex2angle_array(error_measured)
   error_vec_pdoa = error_measured_angle[np.triu_indices(N, k=1)]
   pdoa_var = pdoa_sigma * pdoa_sigma

   time_array = get_time_array(loc_tag, loc_anchor, 0)
   tdoa_ref = time2tdoa(time_array)   

   error_measured = tdoa_measured - tdoa_ref
   error_vec_tdoa = error_measured[np.triu_indices(N, k=1)]
   tdoa_var = tdoa_sigma * tdoa_sigma
   
   cov_matrix = np.diag([tdoa_var] * M + [pdoa_var] * M)
   
   ### Multivariable Normal Random Variable
   mnrv = stats.multivariate_normal([0] * (M * 2), cov_matrix,
                                    allow_singular=True)

   error_vec = np.concatenate([error_vec_tdoa, error_vec_pdoa], 0)
   l = mnrv.pdf(error_vec)

   return l




def estimate_position(loc_tag, loc_anchor, tdoa, pdoa,
                     sigma_time, sigma_phase):
   global SIDELENGTH
   global n_jobs

   eval_matrix = np.zeros((SIDELENGTH + 1, SIDELENGTH + 1))

   max_index = (SIDELENGTH + 1) ** 2

   results = Parallel(n_jobs=n_jobs, verbose=10)(
      delayed(wrap_calc_likelihood_joint)(
         i,loc_anchor,tdoa, pdoa, sigma_time, sigma_phase
      ) for i in range(max_index)
   )

   print("np.argmax(results)")
   index = np.argmax(results)
   print("%d, %d, %d" % (index, index % (SIDELENGTH + 1), index // (SIDELENGTH + 1)))

   return (index % (SIDELENGTH + 1), index // (SIDELENGTH + 1))


def estimate_position_tracker_offset(loc_anchor, tdoa, pdoa,
                                     sigma_time, sigma_phase,
                                     x_offset, y_offset):
   global SIDELENGTH
   global n_jobs

   eval_matrix = np.zeros((SIDELENGTH + 1, SIDELENGTH + 1))

   max_index = (SIDELENGTH + 1) ** 2

   results = Parallel(n_jobs=n_jobs, verbose=10)(
      delayed(wrap_calc_likelihood_joint_tracker_offset)(
          i,loc_anchor,tdoa, pdoa, sigma_time, sigma_phase, x_offset, y_offset
      ) for i in range(max_index)
   )

   print("np.argmax(results)")
   index = np.argmax(results)
   x = index % (SIDELENGTH + 1) + x_offset
   y = ((index // (SIDELENGTH + 1)) + y_offset)
   print("%d, %d, %d" % (index, x, y))

   return (x, y)





def estimate_position_tracker_offset_only_pdoa(loc_anchor, pdoa, sigma_phase,
                                               x_offset, y_offset):
   global SIDELENGTH
   global n_jobs

   eval_matrix = np.zeros((SIDELENGTH + 1, SIDELENGTH + 1))

   max_index = (SIDELENGTH + 1) ** 2

   results = Parallel(n_jobs=n_jobs, verbose=10)(
      delayed(wrap_calc_likelihood_joint_tracker_offset_only_pdoa)(
          i,loc_anchor, pdoa, sigma_phase, x_offset, y_offset
      ) for i in range(max_index)
   )

   print("np.argmax(results)")
   index = np.argmax(results)
   x = index % (SIDELENGTH + 1) + x_offset
   y = -1 * ((index // (SIDELENGTH + 1)) + y_offset)
   print("%d, %d, %d" % (index, x, y))

   return (x, y)



def estimate_position_tracker(loc_anchor, tdoa, pdoa,
                              sigma_time, sigma_phase):
   global SIDELENGTH
   global n_jobs

   eval_matrix = np.zeros((SIDELENGTH + 1, SIDELENGTH + 1))

   max_index = (SIDELENGTH + 1) ** 2

   results = Parallel(n_jobs=n_jobs, verbose=10)(
      delayed(wrap_calc_likelihood_joint_tracker)(
         i,loc_anchor,tdoa, pdoa, sigma_time, sigma_phase
      ) for i in range(max_index)
   )

   print("np.argmax(results)")
   index = np.argmax(results)
   x = index % (SIDELENGTH + 1) - 1000
   y = -1 * (index // (SIDELENGTH + 1))
   print("%d, %d, %d" % (index, x, y))

   return (x, y)



def eval_dpd_resolution(x_tag, y_tag, anchors, sigma_phase, sigma_time):
    global SIDELENGTH
    length = SIDELENGTH
    x_anchor1 = anchors[0][0]
    x_anchor2 = anchors[1][0]    
    
    loc_tag = np.array((x_tag, y_tag))
    loc_anchor = np.matrix(anchors)

    time_array = get_time_array(loc_tag, loc_anchor, sigma_time)
    tdoa = time2tdoa(time_array)
    
    phase_array = get_phase_array(loc_tag, loc_anchor, sigma_phase)
    pdoa = phase2pdoa(phase_array)
    
    ret = estimate_position(loc_tag, loc_anchor, tdoa, pdoa, sigma_time,
                           sigma_phase)

    return ret





def eval_dpd_resolution_tracker(x_tag, y_tag, anchors, sigma_phase, sigma_time):
    global SIDELENGTH
    length = SIDELENGTH
    
    loc_tag = np.array((x_tag, y_tag))
    loc_anchor = np.matrix(anchors)

    time_array = get_time_array(loc_tag, loc_anchor, sigma_time)
    tdoa = time2tdoa(time_array)
    print("loc_tag")
    print(loc_tag)
    print("loc_anchor")
    print(loc_anchor)
    print("tdoa")
    print(tdoa)
    
    phase_array = get_phase_array(loc_tag, loc_anchor, sigma_phase)
    pdoa = phase2pdoa(phase_array)
    
    ret = estimate_position_tracker(loc_tag, loc_anchor, tdoa, pdoa, sigma_time,
                                    sigma_phase)

    return ret


def eval_dpd_resolution_tracker_zero(x_tag, y_tag, anchors, sigma_phase, sigma_time):
    global SIDELENGTH
    length = SIDELENGTH
    x_anchor1 = anchors[0][0]
    x_anchor2 = anchors[1][0]    
    
    loc_tag = np.array((x_tag, y_tag))
    loc_anchor = np.matrix(anchors)

    time_array = get_time_array(loc_tag, loc_anchor, 0)
    tdoa = time2tdoa(time_array)
    print("loc_tag")
    print(loc_tag)
    print("loc_anchor")
    print(loc_anchor)
    print("tdoa")
    print(tdoa)
    
    phase_array = get_phase_array(loc_tag, loc_anchor, 0)
    pdoa = phase2pdoa(phase_array)
    
    ret = estimate_position_tracker(loc_tag, loc_anchor, tdoa, pdoa, sigma_time,
                                    sigma_phase)

    return ret


def wrap_calc_likelihood_joint(index, loc_anchor, tdoa, pdoa,
                               sigma_time, sigma_phase):
   global SIDELENGTH
   x_tag = index % (SIDELENGTH + 1)
   y_tag = index // (SIDELENGTH + 1)
   loc_tag = np.array((x_tag, y_tag))
   ret = calc_likelihood_joint(loc_tag, loc_anchor, tdoa, pdoa,
                                sigma_time, sigma_phase)
   return ret


def wrap_calc_likelihood_joint_tracker(index, loc_anchor, tdoa, pdoa,
                                       sigma_time, sigma_phase):
   global SIDELENGTH
   x_tag = index % (SIDELENGTH + 1) - 1000
   y_tag = -1 * (index // (SIDELENGTH + 1))
   loc_tag = np.array((x_tag, y_tag))
   ret = calc_likelihood_joint(loc_tag, loc_anchor, tdoa, pdoa,
                                sigma_time, sigma_phase)
   return ret

def wrap_calc_likelihood_joint_tracker_offset(index, loc_anchor, tdoa, pdoa,
                                       sigma_time, sigma_phase,
                                       x_offset, y_offset):
   global SIDELENGTH
   x_tag = index % (SIDELENGTH + 1) + x_offset
   y_tag = ((index // (SIDELENGTH + 1)) + y_offset)
   loc_tag = np.array((x_tag, y_tag))
   ret = calc_likelihood_joint(loc_tag, loc_anchor, tdoa, pdoa,
                                sigma_time, sigma_phase)
   return ret


def wrap_calc_likelihood_joint_tracker_offset_only_pdoa(index, loc_anchor, pdoa,
                                       sigma_phase, x_offset, y_offset):
   global SIDELENGTH
   x_tag = index % (SIDELENGTH + 1) + x_offset
   y_tag = ((index // (SIDELENGTH + 1)) + y_offset)
   loc_tag = np.array((x_tag, y_tag))
   ret = calc_likelihood_pdoa(loc_tag, loc_anchor, pdoa, sigma_phase)
   return ret




def calc_likelihood_tdoa(loc_tag, loc_anchor, tdoa_measured, tdoa_sigma):
   N = loc_anchor.shape[0]
   M = (N  * (N - 1)) // 2

   time_array = get_time_array(loc_tag, loc_anchor, 0)
   tdoa_ref = time2tdoa(time_array)   
   error_measured = tdoa_measured - tdoa_ref
   error_vec = error_measured[np.triu_indices(N, k=1)]
   tdoa_var = tdoa_sigma * tdoa_sigma
   cov_matrix = np.diag([tdoa_var] * M)
   mnrv = stats.multivariate_normal([0] * M, cov_matrix,
                                    allow_singular=True)
   l = mnrv.pdf(error_vec)

   return l


def calc_likelihood_pdoa(loc_tag, loc_anchor, pdoa_measured, pdoa_sigma):
   N = loc_anchor.shape[0]
   M = (N  * (N - 1)) // 2

   phase_array = get_phase_array(loc_tag, loc_anchor, 0)
   pdoa_ref = phase2pdoa(phase_array)
   
   error_measured = pdoa_measured / pdoa_ref
   error_measured_angle = complex2angle_array(error_measured)
   error_vec = error_measured_angle[np.triu_indices(N, k=1)]
   pdoa_var = pdoa_sigma * pdoa_sigma
   cov_matrix = np.diag([pdoa_var] * M)
   mnrv = stats.multivariate_normal([0] * M, cov_matrix,
                                    allow_singular=True)
   l = mnrv.pdf(error_vec)

   return l



def create_grid(loc_offset, length, resolution):
   tmp_array = np.arange(- length, length, resolution)

   X, Y = np.meshgrid(tmp_array, tmp_array)
   X += loc_offset[0]
   Y += loc_offset[1]
   grid = np.dstack((X, Y))

   return grid

def multivariate_normal_pdf_matrix(x, mean, covariance):
   k = mean.shape[1]
   norm_const = 1.0 / (np.sqrt((2 * np.pi) ** k * np.linalg.det(covariance[0])))
   x_mu = np.matrix(x - mean)
   inv_covmat = np.linalg.inv(covariance)
   inv_covmat = np.tile(inv_covmat, (len(x_mu), 1, 1))
   inner_exp = np.exp(-0.5 * ((x_mu[:, None] @ inv_covmat)[:, None] @ x_mu[..., None])).T
   return norm_const * np.squeeze(np.asarray(inner_exp))


def compute_joint_prob_matrix(tx_loc, rx_loc, tdoa_measured, tdoa_var, pdoa_measured, pdoa_var, wavelen, bcal_beta=None, bcal_gamma=None):
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
   bcal_beta = scaling bias wrt distance
   bcal_gamma = exponential bias wrt distance
   """

   num_rx = 6
   tx_loc = np.array(tx_loc)
   num_locs = len(tx_loc)
   indices = np.triu_indices(num_rx, k=1)

   pdoa_exp = compute_pdoa_matrix(tx_loc, rx_loc, wavelen, bcal_beta, bcal_gamma)
   measurement_error = np.angle(np.exp(1j*(np.angle(pdoa_measured[None]) - pdoa_exp)))
   
   error_vec_pdoa = measurement_error[:, indices[0], indices[1]]

   tdoa_exp = compute_tdoa_matrix(tx_loc, rx_loc)
   measurement_error = (tdoa_measured[None] - (tdoa_exp * 1e9))
   error_vec_tdoa = measurement_error[:, indices[0], indices[1]]
   combined_error = np.hstack((error_vec_tdoa, error_vec_pdoa))

   sigma = np.diag([tdoa_var] * error_vec_tdoa.shape[1] + [pdoa_var] * error_vec_pdoa.shape[1])

   mean = np.zeros(combined_error.shape)
   prob = multivariate_normal_pdf_matrix(combined_error, mean, sigma[None])

   return prob



def compute_pdoa_matrix(tx_loc, rx_loc, wavelen, bcal_beta=None, bcal_gamma=None):
   distances = np.linalg.norm(tx_loc[:, None] - rx_loc[None], axis=2)
   phases = 2 * np.pi / wavelen * distances
   if bcal_beta is not None:
       phases -= (bcal_beta[None] * (distances ** bcal_gamma[None])) * np.pi / 180
   pdoa_matrix = np.angle(np.exp(1j * (phases[..., None] - phases[:, None])))
   return pdoa_matrix


def compute_tdoa_matrix(tx_loc, rx_loc):
   times = np.linalg.norm(tx_loc[:, None] - rx_loc[None], axis=2) / scipy.constants.c
   tdoa_matrix = times[..., None] - times[:, None]

   return tdoa_matrix



def estimate_position_matrix(loc_anchor, tdoa, pdoa,
                             sigma_time, sigma_phase, 
                             x_answer, y_answer,
                             bcal_beta=None, bcal_gamma=None):
   tic = time.time()
   tx_loc = create_grid([x_answer, y_answer], 1000, 2.0)
   print(time.time() - tic)
   ret = compute_joint_prob_matrix(tx_loc.reshape(-1, 2), loc_anchor, tdoa, sigma_time ** 2, pdoa, sigma_phase ** 2, LAMBDA, bcal_beta, bcal_gamma)
   ret = ret.reshape((tx_loc.shape[0], tx_loc.shape[1]))
   print(time.time() - tic)

   index = np.unravel_index(np.argmax(ret), ret.shape)
   pred = tx_loc[index]

   return pred[0], pred[1]



def estimate_position_matrix_wide(loc_anchor, tdoa, pdoa,
                             sigma_time, sigma_phase, 
                             x_answer, y_answer,
                             bcal_beta=None, bcal_gamma=None):
   tic = time.time()
   tx_loc = create_grid([x_answer, y_answer], 5000, 10.0)
   print(time.time() - tic)
   ret = compute_joint_prob_matrix(tx_loc.reshape(-1, 2), loc_anchor, tdoa, sigma_time ** 2, pdoa, sigma_phase ** 2, LAMBDA, bcal_beta, bcal_gamma)
   ret = ret.reshape((tx_loc.shape[0], tx_loc.shape[1]))
   print(time.time() - tic)

   index = np.unravel_index(np.argmax(ret), ret.shape)
   pred = tx_loc[index]

   return pred[0], pred[1]



def eval_dpd_resolution_matrix(x_tag, y_tag, anchors, sigma_phase, sigma_time):
    x_anchor1 = anchors[0][0]
    x_anchor2 = anchors[1][0]    
    
    loc_tag = np.array((x_tag, y_tag))
    loc_anchor = np.array(anchors)

    time_array = get_time_array(loc_tag, loc_anchor, sigma_time)

    tdoa = time2tdoa(time_array)

    phase_array = get_phase_array(loc_tag, loc_anchor, sigma_phase)    
    pdoa = phase2pdoa(phase_array)

    
    sigma_phase = sigma_phase * np.pi/180
    ret = estimate_position_matrix(loc_anchor, tdoa, pdoa, sigma_time,
                                   sigma_phase, x_tag, y_tag, None, None)

    return ret



def eval_dpd_resolution_matrix_wide(x_tag, y_tag, anchors, sigma_phase, sigma_time):
    x_anchor1 = anchors[0][0]
    x_anchor2 = anchors[1][0]    
    
    loc_tag = np.array((x_tag, y_tag))
    loc_anchor = np.array(anchors)

    time_array = get_time_array(loc_tag, loc_anchor, sigma_time)

    tdoa = time2tdoa(time_array)

    phase_array = get_phase_array(loc_tag, loc_anchor, sigma_phase)    
    pdoa = phase2pdoa(phase_array)

    
    sigma_phase = sigma_phase * np.pi/180
    ret = estimate_position_matrix_wide(loc_anchor, tdoa, pdoa, sigma_time,
                                        sigma_phase, x_tag, y_tag, None, None)

    return ret


