import numpy as np
import tensorflow as tf

class VoiceHandler:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.num_train_example = len(train_data)

        self.test_data = test_data
        self.test_labels = test_labels
        self.num_test_example = len(test_data)

        self.whiten = False
        self.training_index = -20
        self.test_index = -20

    def do_whiten(self):
        self.whiten = True
        train_len = len(self.train_data)
        data_to_be_whitened = np.copy(self.train_data)

        mean = np.sum(data_to_be_whitened, axis=0) / train_len
        mean = np.tile(mean, train_len)
        mean = np.reshape(mean, (train_len, 784))
        
        centered_data = data_to_be_whitened - mean
        covariance = np.dot(centered_data.T, centered_data) / train_len
        U, S, V = np.linalg.svd(covariance)
        epsilon = 1e-5
        lambda_square = np.diag(1./np.sqrt(S+epsilon))
        self.whitening_mat = np.dot(np.dot(U, lambda_square), V)
        self.whitened_train_data = np.dot(centered_data, self.whitening_mat)

        data_to_be_whitened = np.copy(self.test_data)
        test_len = len(self.test_data)

        mean = np.sum(data_to_be_whitened, axis=0) / test_len
        mean = np.tile(mean, test_len)
        mean = np.reshape(mean, (test_len, 784))
        centered_data = data_to_be_whitened - mean
        self.whitened_test_data = np.dot(centered_data, self.whitening_mat)

    def next_batch(self, batch_size, type='train'):
        if type == 'train':
            if self.whiten:
                operand = self.whitened_train_data
            else:
                operand = self.train_data

            operand_bis = self.train_labels

            self.training_index = \
                (batch_size + self.training_index) % len(self.train_data)
            index = self.training_index
            number = len(self.train_data)

        elif type == 'test':
            if self.whiten:
                operand = self.whitened_test_data
            else:
                operand = self.test_data

            operand_bis = self.test_labels

            self.test_index = (
                batch_size + self.test_index) % len(self.test_data)
            index = self.test_index
            number = len(self.test_data)

        if index + batch_size > number:
            part1 = operand[index:, :]
            part2 = operand[:(index + batch_size) % number, :]
            result = np.concatenate([part1, part2])
            part1 = operand_bis[index:, :]
            part2 = operand_bis[:(index + batch_size) % number, :]
            result_bis = np.concatenate([part1, part2])
        else:
            result = operand[index:index + batch_size, :]
            result_bis = operand_bis[index:index + batch_size, :]
        return result, result_bis


