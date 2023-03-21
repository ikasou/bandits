import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.linear_model
import sklearn.neural_network
import scipy as sp
import random
from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        te = time()
        args[0]._runtime += te-ts
        return result
    return wrap

class Bandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.actions = np.zeros((0,), dtype=np.int32)
        self.rewards = np.zeros((0,), dtype=np.float32)
        self._runtime = 0

    @timing
    def pick(self, X):
        return random.randrange(0, self.n_arms)
    
    @timing
    def update(self, X, a, r):
        self.actions = np.append(self.actions, a)
        self.rewards = np.append(self.rewards, r)

    def pick_update(self, X, y, freward, *args):
        arm = min(2, max(0, self.pick(X)))
        r = freward(arm, y)
        self.update(X, arm, r, *args)


class FixedBandit(Bandit):
    def __init__(self, n_arms, fixed_arm):
        super().__init__(n_arms)
        self.fixed_arm = fixed_arm

    @timing
    def pick(self, X):
        return self.fixed_arm

class LinUCBBandit(Bandit):
    def __init__(self, n_arms=3, n_features=8, alpha=1, reg_lambda = 1):
        '''
        :param: alpha is the UCB confidence parameter
        :param: reg_lambda is the L1 norm Lagrange multiplier
        '''
        super().__init__(n_arms)
        self.n_features = n_features
        self.alpha = alpha
        self.reg_lambda = reg_lambda
        self.A = reg_lambda*np.eye(n_features)[None].repeat(n_arms, 0)
        self.b = np.zeros([n_arms, n_features])
    
    @timing
    def pick(self, X: np.ndarray):
        '''
        :param: X is context and can be reshaped into (n_arms, n_features, 1)
        '''
        x = X.reshape(-1, self.n_features, 1)
        xT = x.swapaxes(1, 2)
        Ainv = np.linalg.pinv(self.A); 
        theta = Ainv @ self.b[..., np.newaxis]
        p = xT @ theta + self.alpha*np.sqrt(xT @ Ainv @ x)
        return np.argmax(p)
    
    @timing
    def update(self, X: np.ndarray, a, r):
        super().update(X, a, r)
        x = X.reshape(-1, self.n_features, 1)
        if x.shape[0] > 1: x = x[a]
        self.A[a] += np.outer(x, x)
        self.b[a] += r*x.reshape(self.n_features)

class OracleOLSBandit(Bandit):
    def __init__(self, n_arms, coeff):
        super().__init__(n_arms)
        self.coeff = coeff

    @timing
    def pick(self, X: np.ndarray):
        pred = (self.coeff @ np.hstack((1, X)))**2
        arm = 0 if pred < 21 else 1 if pred < 49 else 2
        return arm
    
class LassoBandit(Bandit):
    def __init__(self, n_arms, n_data, n_features, data, y, 
                 h = 0.25, q = 10, lambda_1 = 0.1, lambda_2_0 = 0.1, 
                 tol = 1e-3, itcpt=True, max_iter=100):
        super().__init__(n_arms)
        self.ndata = n_data
        self.n_features = n_features
        self.data = data
        self.y = y
        self.q = q
        self.h = h               # pstar = (1-h sqrt(2))^2
        self.l_1  = lambda_1/2   # pho0^2 pstar h / (64 s0 xmax)
        self.l_20 = lambda_2_0/2 # (ph0^2/2/s0)sqrt(1/(pstar C1)) 
        K = n_arms               # C1 = phi0^4/(512 so^2 sigma^2 xmax^2)
        self.beta_T = [ sklearn.linear_model.Lasso(alpha=self.l_1, 
                    fit_intercept=itcpt, max_iter=max_iter, tol=tol) for _ in range(n_arms) ]
        self.beta_S = [ sklearn.linear_model.Lasso(alpha=self.l_20, 
                    fit_intercept=itcpt, max_iter=max_iter, tol=tol) for _ in range(n_arms) ]
        self.T = [ [ (2**n-1)*K*q + j 
                for n in range(int(np.log(n_data/q/(K+i))/np.log(2))+1+1) 
                for j in range(q*(i-1)+1, q*i+1) if (2**n-1)*K*q + j <= n_data
            ] for i in range(1, K+1) ]
        self.Ts = [ np.array([], dtype=int) for _ in range(K) ]  # forced sample times
        self.S  = [ np.array([], dtype=int) for _ in range(K) ]
        self.RT = [ np.array([]) for _ in range(K) ]  # rewards
        self.RS = [ np.array([]) for _ in range(K) ]
        self.t = 1    
    
    @timing
    def pick(self, X: np.ndarray):    
        self.forced_update = False
        for a in range(self.n_arms):
            if self.t in self.T[a]:
                self.forced_update = True
                return a
        pred_T = [ self.beta_T[j].predict(X[None]) for j in range(self.n_arms) ]
        Kest = [ i for i in range(self.n_arms) if pred_T[i] >= np.max(pred_T) - self.h/2 ]
        arm = Kest[ np.argmax([ self.beta_S[i].predict(X[None]) for i in Kest ]) ]
        return arm

    @timing 
    def update(self, X: np.ndarray, a, r, tind):
        super().update(X, a, r)
        if self.forced_update:
            self.Ts[a] = np.append(self.Ts[a], tind)
            self.RT[a] = np.append(self.RT[a], r)
            self.beta_T[a].fit(self.data[self.Ts[a]], self.RT[a])
        self.RS[a] = np.append(self.RS[a], r)
        self.S[a] = np.append(self.S[a], tind)              
        self.t += 1
        self.l_2t = self.l_20*np.sqrt((np.log(self.t) + np.log(self.n_features*self.n_arms))/self.t)
        for i, c in enumerate(self.beta_S): 
            if type(self) is not LogisticBandit:
                c.set_params(alpha = self.l_2t)
            if self.S[i].any():
                self.beta_S[a].fit(self.data[self.S[a]], self.RS[a])

# Specific version mirroring Lasso Bandit
class OLSBandit(LassoBandit):
    def __init__(self, n_arms, n_data, n_features, data, y, h = 0.25, q = 10, lambda_1 = 0.1, lambda_2 = 0.1) :
        super().__init__(n_arms, n_data, n_features, data, y, h, q, lambda_1, lambda_2)
        self.beta_T = [ sklearn.linear_model.Ridge(alpha=lambda_1) for _ in range(n_arms) ]
        self.beta_S = [ sklearn.linear_model.Ridge(alpha=lambda_2) for _ in range(n_arms) ] 

class MyLogisticRegression(sklearn.linear_model.LogisticRegression):
    '''
        Doesn't complain for single class fitting
    '''
    def __init__(self):#, *args, **kwargs):
        self._single_class_label = None
        super().__init__()#*args, **kwargs)

    @staticmethod
    def _has_only_one_class(y):
        return len(np.unique(y)) == 1

    def _fitted_on_single_class(self):
        return self._single_class_label is not None

    def fit(self, X, y):
        if self._has_only_one_class(y):
            self._single_class_label = y[0]
        else:
            self._single_class_label = None
            super().fit(X, y)
        return self

    def predict(self, X):
        if self._fitted_on_single_class():
            return np.array(self._single_class_label)
        else:
            return super().predict(X)
        
# Specific version mirroring Lasso Bandit        
class LogisticBandit(LassoBandit):
    def __init__(self, n_arms, n_data, n_features, data, y):
        super().__init__(n_arms, n_data, n_features, data, y)
        self.beta_T = [ MyLogisticRegression() for _ in range(n_arms) ]
        self.beta_S = [ MyLogisticRegression() for _ in range(n_arms) ] 

class OnlineOLSBandit(Bandit):
    def __init__(self, n_arms, n_data, n_features, data, y):
        super().__init__(n_arms)
        self.data = data
        self.y = y
        self.Ts = []
        self.model = sklearn.linear_model.LinearRegression()
        self.model.fit(np.zeros((1, n_features)), [0])
        
    def pick(self, X: np.ndarray):
        return round(self.model.predict(X[None]).item())
    
    def update(self, X: np.ndarray, a, r, ts):
        super().update(X, a, r)
        self.Ts.append(ts)
        self.model.fit(self.data[self.Ts], self.y[self.Ts])
        
class OnlineLogisticRegressionBandit(OnlineOLSBandit):
    def __init__(self, n_arms, n_data, n_features, data, y):
        super().__init__(n_arms, n_data, n_features, data, y)
        self.data = data
        self.y = y
        self.Ts = []
        self.model = MyLogisticRegression()
        self.model.fit(np.zeros((1, n_features)), [0])        
        
class LinearTSBandit(Bandit):
    def __init__(self, n_arms, n_features, R=1):
        super().__init__(n_arms)
        self.n_features = n_features
        self.R = R
        self.vsq = R**2  
        #vsq = (Rsq**2)*(9*self.n_features*np.log((1+t)/delta))
        # B: precision, B_inv: covariance
        self.B = np.eye(n_features)[None].repeat(n_arms, 0)
        self.ba_r = np.zeros((n_arms, n_features, 1))
        self.B_inv = np.copy(self.B)
        self.mean = np.zeros((n_arms, n_features, 1))

    @timing
    def pick(self, X: np.ndarray):
        theta = np.array([ 
            sp.stats.multivariate_normal.rvs(mean=self.mean[i].squeeze(), 
            cov=self.vsq*self.B_inv[i]) for i in range(self.n_arms) ])
        p = theta @ X
        return np.argmax(p)
    
    @timing
    def update(self, X: np.ndarray, a, r):
        super().update(X, a, r)
        self.B[a] += np.outer(X, X)
        self.ba_r[a] += X[:, None] * r
        self.B_inv[a] = np.linalg.pinv(self.B[a]) 
        self.mean[a] = self.B_inv[a] @ self.ba_r[a]


class LinearESBandit(Bandit):
    def __init__(self, n_arms, n_features, R=1, M=100):
        super().__init__(n_arms)
        self.n_features = n_features
        self.R = R
        self.vsq = R**2  
        # B: precision, B_inv: covariance
        self.B = np.eye(n_features)[None, None].repeat(self.n_arms, 0)
        self.B_inv = np.copy(self.B) 
        self.M = M # number of models of ensemble 
        self.ensemble_thetas = np.zeros((self.n_arms, self.M, self.n_features, 1))

    @timing
    def pick(self, X: np.ndarray, ms=None):
        models = ms or np.random.randint(0, self.M, size=self.n_arms)
        thetas = np.array([ self.ensemble_thetas[i, ms or models[i], :, :] 
                           for i in range(self.n_arms) ]).squeeze(2)
        p = thetas @ X.T
        return np.argmax(np.diag(p)) if np.min(p.shape)>1 else np.argmax(p)


    @timing
    def update(self, X: np.ndarray, a, r):
        super().update(X, a, r)
        stdnorm = np.random.randn(self.M, 1, 1)
        self.ensemble_thetas[a] = self.B[a] @ self.ensemble_thetas[a]
        self.ensemble_thetas[a] += X[..., None]*(r/self.vsq + stdnorm)
        if np.min(X.shape) > 1:
            self.B[a] += np.array([np.outer(X[i], X[i]) 
                        for i in range(self.M)]).mean(0)/self.vsq 
        else:
            self.B[a] += np.outer(X, X)/self.vsq  
        self.B_inv[a] = np.linalg.pinv(self.B[a]) 
        self.ensemble_thetas[a] = self.B_inv[a] @ self.ensemble_thetas[a]

 
class NeuralLinearESBandit(LinearESBandit):
    def __init__(self, n_arms, n_features, n_neural_features, R=1, M=10, epochs=4, l2=.1, device='cpu'):
        super().__init__(n_arms, n_features, R=1, M=10)
        self.t = 1
        self.n_neural_features = n_neural_features
        self.l2 = l2
        self.device = device
        self.data =  torch.Tensor().to(self.device) 
        self.y = [ torch.Tensor().to(self.device) for _ in range(self.M) ] 
        self.n_epochs = epochs
        self.batch_size = 32  
        self.loss_fn = nn.MSELoss()  
        self.models = [ 
            nn.Sequential(
                nn.Linear(n_neural_features, 48),
                nn.ReLU(),
                nn.Linear(48, 12),
                nn.ReLU(),
                nn.Linear(12, n_features),
                nn.ReLU(),
                nn.Linear(n_features, n_arms)
            ).to(self.device)
            for _ in range(self.M) ]
        self.optimizers = [ # [ 
            optim.Adam(self.models[m].parameters(),  
            lr=0.0001, weight_decay=l2) # L2 regularization       
            for m in range(self.M) ] 

    @timing
    def _train(self):    
        batch_starts = torch.arange(0, len(self.data), self.batch_size)  
        for m, mdl in enumerate(self.models): 
            mdl.train()
            for _ in range(self.n_epochs):
                for start in batch_starts:
                    X_batch = self.data[start: start + self.batch_size].squeeze(1) 
                    ya_batch = self.y[m][start: start + self.batch_size]
                    y_batch, a_batch = ya_batch[:, 0:1], ya_batch[:, 1:2].long() 
                    y_pred = torch.gather(mdl(X_batch), 1, a_batch) 
                    loss = self.loss_fn(y_pred, y_batch)/self.vsq # + L2 regul via Adam
                    self.optimizers[m].zero_grad() 
                    loss.backward()
                    self.optimizers[m].step() 
            
    @timing  
    def _eval(self, X: np.ndarray):
        x = torch.Tensor(X).to(device=self.device)
        ret = [ [] for _ in range(self.M) ] 
        with torch.no_grad():
            for m, mdl in enumerate(self.models):        
                mdl.eval()
                ret[m].append(mdl(x).cpu().numpy())
        return ret
    
    @timing
    def _score(self, X, y):
        pred = self._eval(X)
        return sum(pred == y)/len(y)
    
    @timing
    def pick(self, X: np.ndarray):
        x = torch.Tensor(X).to(device=self.device)
        ms = np.random.randint(0, self.M)
        with torch.no_grad():
            # drop last 2 layers, only use NN for feature extraction
            X_nn_feat_values = torch.nn.Sequential(*list(self.models[ms].children())[:-2])(x).cpu().numpy() 
        return super().pick(X_nn_feat_values, ms) 
      
    @timing
    def update(self, X: np.ndarray, a, r):
        new_sample = torch.Tensor(X).to(self.device).reshape(-1, self.n_neural_features)
        self.data = torch.cat((self.data, new_sample)) 
        rnorms = np.random.randn(self.M)*self.vsq
        new_ys = [ torch.Tensor([[r + z, a]]).to(self.device) for z in rnorms ]
        for i, ny in enumerate(new_ys):
            self.y[i] = torch.cat((self.y[i], ny)) 
        
        if self.t % 1 == 0:
            self._train()
        self.t += 1
        
        with torch.no_grad():
            # drop last 2 layers, only use NN for feature extraction
            X_nn_feat_values = np.array([ 
                torch.nn.Sequential(*list(self.models[m].children())[:-2])(new_sample).cpu().numpy() # [a]
            for m in range(self.M) ])

        super().update(X_nn_feat_values.squeeze(1), a, r)


class OracleNeuralBandit(Bandit):
    def __init__(self, n_arms, n_features, data, y):
        super().__init__(n_arms)
        self.n_features = n_features
        self.data = data
        self.y = y
        self.train_data, self.test_data, self.train_y, self.test_y = \
            train_test_split(data, y)
        self.model = sklearn.neural_network.MLPClassifier(
            hidden_layer_sizes=(64,64),activation="relu",
            max_iter=1000)
        self._train()
        
    def _train(self):
        self.model.fit(self.data, self.y)

    def _eval(self, X=None):
        x = X if X is not None else self.data
        return self.model.predict(x)
    
    def _score(self, X=None, y=None):
        x = X if X is not None else self.data
        y = y if y is not None else self.y
        return self.model.predict(x)

    @timing
    def pick(self, X):
        return round(self.model.predict(X[None]).item())
    

class OnlineNeuralBandit(Bandit):
    def __init__(self, n_arms, n_features, data, y, epochs=100):
        super().__init__(n_arms)
        self.n_features = n_features
        self.data = torch.Tensor(data)
        self.y = torch.Tensor(y)
        self.Ts = []
        self.model = sklearn.neural_network.MLPClassifier(
            hidden_layer_sizes=(64,64),activation="relu",
            random_state=1, max_iter=2)
        self.model.fit(np.zeros((1, n_features)), np.array([0]))

    @timing  
    def _eval(self, X: np.ndarray):
        if len(X):
            return self.model.predict(X)
        else:
            return self.model.predict(self.data[self.Ts])
    
    @timing
    def _score(self, X, y):
        if len(X) and len(y):
            return self.model.score(X, y)
        else:
            return self.model.score(self.data[self.Ts], self.y[self.Ts])
    
    @timing
    def pick(self, X):
        return self.model.predict(X[None])
        
    def update(self, X: np.ndarray, a, r, ts):
        super().update(X, a, r)
        self.Ts.append(ts)
        self.model.fit(self.data[self.Ts], self.y[self.Ts])


 