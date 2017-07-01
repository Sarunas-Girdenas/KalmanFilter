import numpy as np

class ConstantGain():
    
    '''Purpose: implement simple constant gain algorithm
    for linear regression.
    For Matlab version see:
    https://github.com/Sarunas-Girdenas/NK_Learning/blob/master/ParallelComputing/CG_Learning.m
    '''

    def __init__(self, gain_parameter,
                 previous_parameters,
                 d_matrix, z_matrix, variable):

        '''Initialize variables
        '''

        self.gain_parameter = gain_parameter
        self.previous_parameters = previous_parameters
        self.d_matrix = d_matrix
        self.z_matrix = z_matrix
        self.variable = variable
        
        return None

    def cg_learning(self):
        '''Purpose: estimate parameters using CG
        '''
        
        self.parameters_out = self.previous_parameters + \
            np.dot(self.gain_parameter*np.dot(np.linalg.inv(self.d_matrix), self.z_matrix),
                  (self.variable - np.dot(self.previous_parameters.transpose(), self.z_matrix)))
        # update moments matrix

        self.d_matrix = self.d_matrix + np.dot(self.gain_parameter,
        (np.dot(self.z_matrix, self.z_matrix.transpose())) - self.d_matrix)

        return self.parameters_out
