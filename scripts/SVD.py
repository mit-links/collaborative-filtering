import csv
import datetime as dt
import json
import math
import os
import pickle
import random
import statistics as stat
from itertools import product

import numpy as np
from numpy import linalg as LA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

nmovies = 1000
nusers = 10000
use_pickled_training_data = True


class Rating:
    def __init__(self, userid, movieid, rating):
        self.uid = userid - 1   # 1-based indexing is to be DESTROYED!
        self.mid = movieid - 1  # 1-based indexing is to be DESTROYED!
        self.rat = float(rating)


class Core:
    """
    Given some fixed parameters and unknowns, calculate predictions.
    """

    def __init__(self, param_config, traindata):
        """
        Constructor
        :param ParameterConfiguration param_config: a ParameterConfiguration object
        :param [Rating] traindata: list of ratings
        """
        self.param_config = param_config
        self.nusers = nusers
        self.nmovies = nmovies
        self.it = param_config.it                   # max number iterations
        self.lrate = param_config.lrate             # learning rate
        self.reg1 = param_config.reg1               # for regularization
        self.reg2 = param_config.reg2               # for reg. of biases
        self.k_svd = param_config.k_svd             # k for svd
        self.n_loops = param_config.n_loops         # number regularized SVD run after each other
        self.thresh = param_config.thresh           # threshold to abort iteration
        self.t = param_config.t                     # parameter for novel_init
        self.traindata = traindata                  # a list of all ratings
        self.train = np.zeros((nusers, nmovies))    # training data in matrix form (unknowns 0)
        self.mask_train = np.zeros((self.nusers, self.nmovies)).astype(int)
        self.mask_unknown = np.ones((self.nusers, self.nmovies)).astype(int)
        self.U = np.zeros((nusers, self.k_svd))     # modified U matrix
        self.V = np.zeros((self.k_svd, nmovies))    # modified V matrix
        self.c = np.zeros(nusers)                   # bias for users
        self.d = np.zeros(nmovies)                  # bias for movies
        self.global_mean = 0.0                      # global mean of ratings
        self.global_var = 0.0                       # global variance of ratings
        self.initialized = False                    # Core object initialized
        self._init_map = {                          # initialization methods
            "all_3": self._initialize_all_3,
            "avg": self._initialize_avg,
            "smart": self._initialize_improved,
            "smart++": self._initialize_novel,
        }
        self._svd_map = {                           # SVD methods
            "simple": self._train_svd_simple,
            "regularized": self._train_svd_regularized,
        }

    def initialize(self, silent=False):
        """
        Initialize Core object.
        """
        if not self.initialized:
            print("Init: method %s" % self.param_config.init)
            for rating in self.traindata:
                self.mask_train[rating.uid, rating.mid] = 1
                self.mask_unknown[rating.uid, rating.mid] = 0
            initializer = self._init_map[self.param_config.init]
            initializer()
            print("Init: done")
            self._make_svd(self.M)
            self.initialized = True
        elif not silent:
            print("Init: Already initialized")

    def fit(self):
        """
        Fit core to provided training data.
        """
        self.initialize(silent=True)
        self._train_all()

    def _initialize_all_3(self):
        """ Init traindata to be all 3 if value unknown. """
        self.M = 3 * np.ones((self.nusers, self.nmovies))
        for rating in self.traindata:  # put read values into matrix and set mean/var
            self.train[rating.uid, rating.mid] = rating.rat
            self.M[rating.uid, rating.mid] = rating.rat

    def _initialize_avg(self):
        """ Init unknowns to be avg rating of the movie. """
        self.M = 3 * np.ones((self.nusers, self.nmovies))
        for rating in self.traindata:  # put read values into matrix and set mean/var
            self.train[rating.uid, rating.mid] = rating.rat
        for mid in range(self.nmovies):
            n = np.sum(self.mask_train[:, mid])  # number of ratings for movie mid
            if n != 0:
                avg = np.sum(self.train[:, mid]) / n  # avg rating for movie mid
                self.M[self.mask_unknown.astype(bool)[:, mid], mid] = avg  # set unknowns

    def _initialize_novel(self):
        """ Init unknowns with novel approach. """
        self._initialize_improved(self.param_config.t)

    def _initialize_improved(self, t=1):
        """ Init unknowns with improved approach. """
        values = []
        for rating in self.traindata:  # put read values into matrix and set mean/var
            self.train[rating.uid, rating.mid] = rating.rat
            values.append(rating.rat)

        self.global_mean = stat.mean(values)
        self.global_var = stat.variance(values, self.global_mean)
        print("reading test data to array done, mean/var %s/%s" %
              (self.global_mean, self.global_var))

        print("calculating missing values...")
        # calculate all the column means (aka the avg score per movie)
        col_mean = np.zeros(self.nmovies)
        avg_user_offset = np.zeros(self.nusers)  # the avg offset per user from the avg of the movie

        no_ratings_per_movie = {}
        no_ratings_per_user = {}

        for i in range(self.nmovies):
            sum = 0
            count = 0
            values = []
            for j in range(self.nusers):
                if self.train[j][i] != 0:
                    sum += self.train[j][i]
                    count += 1
                    values.append(self.train[j][i])
            no_ratings_per_movie[i] = count
            if len(values) < 2:  # variance needs at least 2 data points
                movie_var = 0
            else:
                movie_var = stat.variance(values)
            # K used to weigh the global mean more if this movie's rating's variance is high
            K = movie_var / self.global_var
            if (K + count) == 0:  # prevent division by 0
                col_mean[i] = self.global_mean
            else:
                col_mean[i] = (self.global_mean * K + sum) / (K + count)

        for i in range(self.nusers):
            sum = 0
            count = 0
            values = []
            for j in range(self.nmovies):
                if self.train[i][j] != 0:  # user actually has a rating
                    sum += (col_mean[j] - self.train[i][j])
                    count += 1
                    values.append(self.train[i][j])
            no_ratings_per_user[i] = count
            if len(values) < 2:  # variance needs at least 2 data points
                user_var = 0
            else:
                user_var = stat.variance(values)
            # K used to weigh the global mean more if this users's rating's variance is high
            K = user_var / self.global_var
            if (K + count) == 0:  # prevent division by 0
                avg_user_offset[i] = 0
            else:
                avg_user_offset[i] = (self.global_mean * K + sum) / (K + count)

        # set missing values of training data to the average of the column
        self.M = np.zeros((self.nusers, self.nmovies))
        for i in range(self.nusers):  # set missing values with column means
            for j in range(self.nmovies):
                # use d to weigh the offset and the movie mean
                d = no_ratings_per_movie[j] / (t * no_ratings_per_user[i] + no_ratings_per_movie[j])

                # if there was no input for this value
                # set it to the avg rating for this movie minus the offset
                if self.train[i][j] == 0:
                    self.M[i][j] = d * col_mean[j] - (1 - d) * avg_user_offset[i]
                else:
                    self.M[i][j] = self.train[i][j]

    def _make_svd(self, to_svd):
        """
        Calculate SVD of to_svd and combine singular values into U,V

        :param numpy.array to_svd: Matrix to decompose.
        """
        print("Simple SVD: decompose with k = %s" % self.k_svd)
        U, s, Vt = LA.svd(to_svd, full_matrices=False)
        print("Simple SVD: done")

        S = np.zeros((self.nusers, self.nmovies))  # fill in with zeroes
        S[:self.nmovies, :self.nmovies] = np.diag(s)
        # truncate matrices to use only k singular values
        U = U[:, :self.k_svd]
        S = S[:self.k_svd, :self.k_svd]
        Vt = Vt[:self.k_svd, :]

        d = np.sqrt(S)
        self.U = np.dot(U, d)
        self.V = np.dot(d, Vt)

    def _train_feature_it(self, k):
        """
        One iteration to train feature k in regularized SVD.

        :param int k: feature to train
        :returns: RMSE difference between the last and the current iteration
        :rtype: float
        """
        sum_sq_errors = 0.0
        n = 0
        for i in range(len(self.traindata)):
            cur_rating = self.traindata[i]
            err = cur_rating.rat - self._predict_it(cur_rating.uid, cur_rating.mid)

            sum_sq_errors += err ** 2
            n += 1

            # update u and v
            u_temp = self.U[cur_rating.uid][k]
            v_temp = self.V[k][cur_rating.mid]
            self.U[cur_rating.uid][k] += self.lrate * (err * v_temp - self.reg1 * u_temp)
            self.V[k][cur_rating.mid] += self.lrate * (err * u_temp - self.reg1 * v_temp)

            # update biases
            c_temp = self.c[cur_rating.uid]
            d_temp = self.d[cur_rating.mid]
            b = self.lrate * (err - self.reg2 * (c_temp + d_temp - self.global_mean))
            self.c[cur_rating.uid] += b
            self.d[cur_rating.mid] += b
        return math.sqrt(sum_sq_errors / n)  # return RMSE

    def _train_feature(self, k):
        """
        Train feature k in regularized SVD.
        :param int k: feature to train
        """
        old_train_err = 100000.0
        print("\tFeature %s: training" % k)
        for i in range(self.it):
            train_err = self._train_feature_it(k)
            if abs(train_err - old_train_err) < self.thresh:
                print("\tFeature %s: aborted after iteration %s" % (k, i))
                break
            old_train_err = train_err

    def _train_svd_simple(self):
        """ SVD already done be initialization. """
        self.M = np.dot(self.U, self.V)

    def _train_svd_regularized(self):
        """ Train regularized SVD for n_loops iterations. """
        # one loop is always necessary
        print("Reg SVD: iteration 0")
        for k in range(self.k_svd):
            self._train_feature(k)
        # per iteration: reconstruct the original matrix by multiplying U and V and start over
        for i in range(1, self.n_loops):
            print("Reg SVD: iteration %s", i)
            new_init = np.dot(self.U, self.V)
            self._make_svd(new_init)
            for k in range(self.k_svd):
                self._train_feature(k)
        self.M = np.dot(self.U, self.V)
        for uid in range(len(self.c)):
            self.M[uid, :] += self.c[uid]
        for mid in range(len(self.d)):
            self.M[:, mid] += self.d[mid]

    def _train_all(self):
        """ Train pipeline. """
        print("Train all: mode %s" % self.param_config.svd)
        self._svd_map[self.param_config.svd]()

        if self.param_config.knn:
            print("Train all: KNN with k_knn %s" % self.param_config.k_knn)
            self._fit_knn(self.param_config.k_knn)
        else:
            print("Train all: KNN skipped")
        print("Train all: done")

    def _fit_knn(self, k):
        """
        Train KNN model for k nearest neighbours.

        :param int k: number of nearest neighbors.
        """
        self.U_, _, _ = LA.svd(self.M, full_matrices=False)
        self.U_ = self.U_[:, :self.k_svd]
        u = normalize(self.U_)
        print("KNN: metric %s" % self.param_config.metric)
        if self.param_config.metric == "cosine":
            nbrs = NearestNeighbors(
                n_neighbors=k, algorithm='auto', n_jobs=-1, metric="euclidean").fit(u)
        else:
            nbrs = NearestNeighbors(
                n_neighbors=k, algorithm='auto', n_jobs=-1, metric=self.param_config.metric).fit(u)
        self.distances, self.indices = nbrs.kneighbors(u)

    def _predict_it(self, uid, mid):
        """
        Prediction for iterations in regularized SVD.

        :param int uid: user id.
        :param int mid: movie id.
        :returns: the predicted rating of user uid for movie mid.
        :rtype: float
        """
        # sum of biases and dot product
        p = self.c[uid] + self.d[mid] + np.dot(self.U[uid, :], self.V[:, mid])
        if p > 5:
            return 5
        elif p < 1:
            return 1
        return p

    def _predict(self, uid, mid):
        """
        Prediction of user uid for movie mid.

        :param int uid: user id.
        :param int mid: movie id.
        :returns: the predicted rating of user uid for movie mid.
        :rtype: float
        """
        p = self.M[uid, mid]
        if hasattr(self, "indices"):  # only if KNN model has been trained
            if self.param_config.metric == "cosine":
                weights = np.dot(self.U_[uid, :], np.transpose(self.U_[self.indices[uid], :]))
                weights = np.divide(weights, LA.norm(self.U_[uid, :]), weights)
                norms = np.array([LA.norm(self.U_[i, :]) for i in self.indices[uid]])
                weights = np.divide(weights, norms, weights)
                weights = np.subtract(1, weights, weights)
            else:
                weights = np.subtract(1, self.distances[uid])
                weights = np.power(weights, 4, weights)
            ratings = self.M[self.indices[uid], mid]
            for i in range(len(ratings)):
                if ratings[i] < 1:
                    ratings[i] = 1
                elif ratings[i] > 5:
                    ratings[i] = 5
            p = ((1 - self.param_config.w_knn) * p +
                 self.param_config.w_knn * (np.dot(ratings, weights) / np.sum(weights)))
        if p > 5:
            return 5
        elif p < 1:
            return 1
        return p

    def predict_all(self, to_predict):
        """
        Predict all ratings for movies in the list.

        :param [(int, int)] to_predict: list of tuples (uid, mid) of indices to predict
        :returns: list of tuples of the form (index, rating) where index is of the form r123_c456
        :rtype [(string, float)]
        """
        output = []
        print("Predict all: start")
        for t in to_predict:
            uid = int(t[0])
            mid = int(t[1])
            # if the value is missing in the training data, compute it
            if self.train[uid, mid] == 0:
                rating = self._predict(uid, mid)
            else:
                rating = self.train[uid, mid]
            out_ind = IO.get_out_ind(uid, mid)
            output.append((out_ind, rating))

        print("Predict all: done")
        return output


class IO:
    """
    Handles reading and writing to/from files
    """

    @staticmethod
    def get_row_ind(ind):  # r123_c456 -> 123 is the row index
        """
        Parse row index.

        :param str ind: row/col index of form rRRR_cCCC.
        :returns: row index
        :rtype: int
        """
        underscore_ind = ind.index("_")
        return int(ind[1:underscore_ind])

    @staticmethod
    def get_col_ind(ind):  # r123_c456 -> 456 is the col index
        """
        Parse column index.

        :param str ind: row/col index of form rRRR_cCCC.
        :returns: column index
        :rtype: int
        """
        underscore_ind = ind.index("_")
        return int(ind[underscore_ind + 2:])

    @staticmethod
    def get_out_ind(r_id, col_id):  # format output in the form r123_c456
        """
        Combine row and column index to out index.

        :param int r_id: row index.
        :param int col_id: column index.
        :returns: out index.
        :rtype: string
        """
        return "r" + str(r_id + 1) + "_c" + str(col_id + 1)

    @staticmethod
    def write_output(name, output):
        """ Write output to file. """
        file_name = os.path.join('..', '..', 'out', name + ".csv")
        with open(file_name, 'wt') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["Id", "Prediction"])  # header

            for out in output:
                ind = out[0]  # the index in the form r123_c456
                value = out[1]
                writer.writerow([ind, float(value)])  # write line
            print(str(len(output)) + " values written")

    @staticmethod
    def read_config_file():
        """ Read config file. """
        with open(os.path.join('..', 'config', 'config.json')) as config_file:
            return json.load(config_file)

    @staticmethod
    def get_indices_to_predict():
        """
        Get indices to be predicted.

        :returns: list of indices to be predicted.
        :rtype: [(int, int)]
        """
        indices_to_predict = []  # a list of tuples of indices to predict
        # get the indices which we need to predict
        with open(os.path.join('..', 'in', 'sampleSubmission.csv'), 'rt') as sample:
            first = True
            reader = csv.reader(sample, delimiter=',')
            for row in reader:
                if first:  # skip header
                    first = False
                    continue
                ind = row[0]
                uid = IO.get_row_ind(ind) - 1
                mid = IO.get_col_ind(ind) - 1
                indices_to_predict.append((uid, mid))  # the index in the form r123_c456
        print(str(len(indices_to_predict)) + " ratings need to be predicted")
        return indices_to_predict

    @staticmethod
    def read_training_data():
        """ Read training data. """
        traindata = []
        with open(os.path.join('..', 'in', 'data_train.csv'), 'rt') as csvfile:
            first = True
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if first:  # skip header
                    first = False
                    continue
                ind = row[0]
                value = row[1]
                uid = IO.get_row_ind(ind)
                mid = IO.get_col_ind(ind)
                traindata.append(Rating(uid, mid, value))  # Rating handles 1-based indexing
        return traindata

    @staticmethod
    def append_to_log(config, error):
        """ Append to log file. """
        with open(os.path.join('..', '..', 'out', 'log.csv'), 'a') as log_file:
            writer = csv.writer(log_file, delimiter='\t')
            now = dt.datetime.now()
            writer.writerow([str(now), str(os.getpid()) + ": " + str(config), str(error)])


class Predictor:
    """
    Predicts the output which can be submitted to kaggle.
    (uses the entire test set and does not give an error estimate)
    """

    def __init__(self, param_config):
        self.param_config = param_config

    def start(self):
        """ Start prediction process. """
        print("Predictor: start with config:\n%s" % str(vars(self.param_config)))
        to_predict = IO.get_indices_to_predict()
        train_data = read_train_data()

        c = Core(self.param_config, train_data)
        c.initialize()
        c.fit()
        output = c.predict_all(to_predict)

        file_name = ("%.19s" % dt.datetime.now()).replace(" ", "_").replace(":", ".")
        print("Predictor: writing output to", file_name)
        IO.write_output(file_name, output)
        config_name = os.path.join('..', '..', 'out', file_name + ".config")
        with open(config_name, 'wt') as config:
            config.write(str(vars(self.param_config)))
        print("Predictor: done. PID %s" % os.getpid())


class ParameterConfiguration:
    """
    this class is filled dynamically while the parameters are read from the config file
    """
    pass


class Parameters:
    def __init__(self, config):
        """
        Creates a parameter object from a json object from the config file
        :param config: the json object with the parameters
        """
        self._parameters = []  # a list of parameter configurations
        self._counter = 0

        keys = []
        value_lists = []

        for key, value in config.items():
            keys.append(key)
            value_lists.append(value)

        for p in product(*value_lists):  # loops over all possible combinations of parameters
            param = ParameterConfiguration()
            for i in range(len(p)):
                # this dynamically adds fields to ParameterConfiguration object
                setattr(param, keys[i], p[i])
            self._parameters.append(param)

    def get_next_parameters(self):
        if self._counter == len(self._parameters):
            return None  # if all parameters have been queried, return None
        else:
            self._counter += 1
            return self._parameters[self._counter - 1]

    def count(self):
        """
        Return number of parameter configurations.

        :returns: number of parameter configurations.
        :rtype: int
        """
        return len(self._parameters)

    def reset_parameters(self):
        self._counter = 0


class Tester:
    """
    Test all possible combinations of some given parameters and calculates the corresponding errors.
    Also uses Predictor to predict the submission data for the best set of parameters.
    """

    def __init__(self, params):
        self._parameters = params
        self.train_data = read_train_data()

    def _calc_error_estimate(self, param_config):
        print("Tester: calculating error estimate...")
        k = int(len(self.train_data) / 5)  # factor to take as test set
        print("Tester: test set size is " + str(k))

        random.shuffle(self.train_data)  # shuffle training data. First k*size elements for testing
        train_data = self.train_data[k:]
        test_data = self.train_data[:k]

        # Fresh data is calculated and nothing saved (because random shuffle).
        core = Core(param_config, train_data)
        core.initialize()
        core.fit()

        to_predict_test = []
        for rating in test_data:
            to_predict_test.append((rating.uid, rating.mid))

        output = core.predict_all(to_predict_test)

        assert len(output) == len(to_predict_test)

        sum_sq_errors = 0.0
        n = 0
        for i in range(len(output)):
            err = test_data[i].rat - output[i][1]  # the second element in the tuple is the rating
            sum_sq_errors += err ** 2
            n += 1

        total_error = math.sqrt(sum_sq_errors / n)  # RMSE

        print("Tester: training error: %s for config: \n%s" % (
            total_error, str(vars(param_config))))
        return total_error

    def start(self):
        """ Start testing process. """
        best_config = None
        best_error = 10000.0
        p = self._parameters.get_next_parameters()
        i = 0
        print("Tester: %s configurations will be tested" % self._parameters.count())
        while p is not None:
            print("Tester: testing config %s:\n%s" % (i, str(vars(p))))
            cur_error = self._calc_error_estimate(p)

            IO.append_to_log(vars(p), cur_error)  # log the config with error and date

            if cur_error < best_error:
                best_config = p
                best_error = cur_error
            p = self._parameters.get_next_parameters()
            i += 1

        print("Tester: writing the submission file for the best output: error %s "
              "pid %s config: \n%s" % (best_error, os.getpid(), str(vars(best_config))))
        print("Tester: done.")
        pred = Predictor(best_config)
        pred.start()


def read_train_data():
    """ Read train data. Use pickle if possible. """
    if use_pickled_training_data:
        try:
            return pickle.load(open("traindata.p", "rb"))
        except FileNotFoundError:
            pass
    data = IO.read_training_data()
    pickle.dump(data, open("traindata.p", "wb"))
    return data
