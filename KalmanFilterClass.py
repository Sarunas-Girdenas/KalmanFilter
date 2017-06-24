import numpy as np

class KalmanFilter():

    """Purpose: run Kalman Filter
    """

    def __init__(self, previous_beta, prior_beta, posterior_beta,
                 previous_p, prior_p, posterior_p, q_mat, h_vec,
                 r_parameter, variable, number_of_state_variables, state_transition):
        
        self.previous_beta = previous_beta
        self.prior_beta = prior_beta
        self.posterior_beta = posterior_beta
        
        self.previous_p = previous_p
        self.prior_p = prior_p
        self.posterior_p = posterior_p

        self.q_mat = q_mat
        self.h_vec = h_vec
        self.r_parameter = r_parameter
        self.variable = variable
        self.number_of_state_variables = number_of_state_variables
        self.state_transition = state_transition

        return None
    
    def kalman_filter_update(self):
        """Purpose: update state (parameters) estimate
        """
        
        prediction_error = self.variable - np.dot(self.prior_beta, self.h_vec)
        self.s_mat = np.dot(np.dot(self.h_vec, self.prior_p), self.h_vec.transpose()) + self.r_parameter
        self.k_vec = np.dot(self.prior_p, self.h_vec.transpose()) / self.s_mat
        self.posterior_beta = self.prior_beta + (self.k_vec * prediction_error)
        self.posterior_p = (np.eye(self.number_of_state_variables) - self.k_vec*self.h_vec)*self.prior_p

        # update the prediction
        self.previous_beta = self.posterior_beta
        self.previous_p = self.posterior_p

        posterior_beta_output = self.posterior_beta

        return posterior_beta_output

    def kalman_filter_predict(self):
        """Purpose: predict using state estimate (parameters)
        """

        self.prior_beta = self.state_transition * self.previous_beta
        self.prior_p = (self.state_transition * self.previous_p) + self.q_mat
        prior_beta_output = self.prior_beta

        return prior_beta_output
