"""
@package Models.py

Provides the wrapper classes for models as building blocks of the ROM classes:
- POD: Wrapper for sklearn-based SVD
- GPR: Base class for Gaussian process regressor
- sklGPR: Wrapper for sklearn.gaussian_process.GaussianProcess (NOT GPR!)
- gpfGPR: Wrapper for GPflow.gpr.GPR
- gpfGPRFITC: Wrapper for GPflow.sgpr.GPRFITC

@author Daning Huang
@date   11/05/2019
"""

# Standard
import time
# Third-party
try:
    import gpflow as GPflow
except ImportError:
    print("MOBO::Models: GPflow not supprted.")
import numpy as np
import scipy as sp
from sklearn.decomposition import TruncatedSVD
# User
try:
    from pyAeroUtils.miscUtils import score
except ImportError:
    from mobo.extern.misc import score
from mobo.extern import gaussian_process
from mobo.src.Sampling import Sampling
from mobo.src.Logging import logging

JITTER = 10 * gaussian_process.MACHINE_EPSILON

# ======================================================================
#         The sklearn-based POD model
# ======================================================================
class POD(logging):
    """
    Defines the POD method with mean shift.
    """
    def __init__(self, **kwargs):
        super(POD, self).__init__(lvl=0)
        self._svd = TruncatedSVD(n_components = kwargs['n_components'],
                                 algorithm    = kwargs['algorithm'])
        self.normalize = kwargs['normalize']

    def fit(self, Y):
        """
        Apply truncated SVD on the input data.
        """
        if self.normalize:
            self.mean_ = np.mean(Y, axis=0)
        else:
            self.mean_ = np.zeros_like(Y[0])
        Yres = Y - self.mean_
        self._svd.fit(Yres)
        self.explained_variance_ratio_ = self._svd.explained_variance_ratio_
        self.components_ = self._svd.components_

    def transform(self, Y):
        """
        Project the input data onto the SVD basis.
        """
        return self._svd.transform(Y - self.mean_)

    def inverse_transform(self, y):
        """
        Convert the SVD components back to full dimensional space.
        """
        return self._svd.inverse_transform(y) + self.mean_

# ======================================================================
#         The base class for GPR models
# ======================================================================
class GPR(logging):
    """
    Defines the basic operations in GPR model.

    Level of details in print-out messages, printLvl=
      -1, nothing
      0, print one line when training starts
      1, add steps of training
      2, add details of hyperparameters
    """
    def __init__(self, **kwargs):
        super(GPR, self).__init__(lvl=0)
        self.def_opt = {
            'corr'      : "squared_exponential",  # The correlation function
            'regr'      : "linear",               # The regression function
            'verbose'   : False,                  # If print out the fitting procedure
            # GPR
            'maxIter'   : 3000,                   # Maximum number of iteration
            'method'    : "BFGS",                 # Algorithm for finding hyperparameters
            'noiseLvl'  : JITTER,                 # Prescribed noise variance
            # sklGPR
            'theta'     : None,                   # [Initial guess, lower limit, upper limit]
                                                  # of hyperparameters
            # GPRFITC
            'numPseudo' : 50,                     # Number of pseudo input points
            'Z'         : None,                   # Initial guess of pseudo inputs
            }
        for _k, _v in self.def_opt.items():
            setattr(self, _k, _v)
        for _k, _v in kwargs.items():
            if _k in self.def_opt:
                setattr(self, _k, _v)
        self.gpType = ''

        if self.corr == "squared_exponential":
            self._corr = CorrSqre()
        elif self.corr == "matern12":
            self._corr = CorrMt12()
        elif self.corr == "matern32":
            self._corr = CorrMt32()
        elif self.corr == "matern52":
            self._corr = CorrMt52()
        else:
            self._corr = None

        if self.regr == "constant":
            self._regr = RegrCnst()
        elif self.regr == "linear":
            self._regr = RegrLinr()
        else:
            self._regr = None

        # If corr and regr are manually implemented,
        # the trained model will be converted to work with user implementation
        self._ifCnv = self._corr and self._regr

    # -----------------------------------------
    # Interface
    # -----------------------------------------
    def fit(self, Xdim, Ydim):
        """
        Fit the model for hyperparameters and interpolation coefficients.
        """
        pass

    def update(self, Xnew, Ynew):
        """
        Update interpolation coefficients with a new sample data.
        """
        try:
            self.update_sample(Xnew, Ynew)
            self.update_coef()
        except ValueError:
            self.printMsg("Failed to update model parameters", -1)

    def update_sample(self, Xnew, Ynew):
        """
        Update the sample data X_/X, Y_.
        """
        _x, _y = self._dmData()
        Xdim = np.vstack([_x, Xnew])
        Ydim = np.vstack([_y, Ynew])
        self._ndData(Xdim, Ydim)
        self.X = self.X_

    def update_coef(self):
        """
        Update interpolation coefficients.
        The data members beta, gamma, C, Ft, G, sigma2 will be updated.
        """
        # Decomposition of a few matrices
        kern, _, _, mshp = self._getCRFun(jac=False)
        R = kern(self.X_, self.X_) + self.noiseLvl * np.eye(len(self.X_))
        try:
            self.C = np.linalg.cholesky(R)
        except np.linalg.linalg.LinAlgError:
            self.printMsg("Models::GPR::update_coef: Failed in Cholesky decomposition, " \
                          "increasing noise level.", -1)
            R += 100.0*self.noiseLvl * np.eye(len(self.X_))
            self.C = np.linalg.cholesky(R)
        H = mshp(self.X_)
        self.Ft = sp.linalg.solve_triangular(self.C, H, lower=True)
        Q, self.G = sp.linalg.qr(self.Ft, mode='economic')

        # Update beta
        Yt = sp.linalg.solve_triangular(self.C, self.Y_, lower=True)
        self.beta = sp.linalg.solve_triangular(self.G, np.dot(Q.T, Yt))

        # Update gamma
        rho = Yt - np.dot(self.Ft, self.beta)
        self.gamma = sp.linalg.solve_triangular(self.C.T, rho)

        # Update sigma2
        sig = self._diag(rho)/len(self.X_)
        self.sigma2 = sig.reshape(-1)

    def predict(self, Xdim, eval_MSE=False, use_lib=True):
        """
        Predict the output w/wo sigma and w/wo 3rd-party lib.
        """
        inp = self._procInp(Xdim)
        if use_lib:
            # Library-provided prediction function
            out = self._predict_lib(inp, eval_MSE)
        else:
            # User-implemented prediction function
            out = self._predict_usr(inp, eval_MSE)
        if eval_MSE:
            return self._dmOut(out[0]), self._dmMSE(out[1])
        return self._dmOut(out)

    def predict_wj(self, Xdim, eval_MSE=False):
        """
        Predict the output w/wo sigma, with jacobian.
        """
        inp = self._procInp(Xdim)
        out = self._predict_wj(inp, eval_MSE)

        if eval_MSE:
            return [self._dmOut(out[0][0]), self._dmJac(out[0][1])], \
              self._dmMSE(out[1][0], jac=out[1][1])
        return self._dmOut(out[0]), self._dmJac(out[1])

    def score(self, Xdim, Ydim):
        """
        Check accuracy of prediction.
        """
        return score(self.predict, Xdim, Ydim.reshape(-1,self.nout), norm=2)

    # -----------------------------------------
    # Core functions
    # -----------------------------------------
    def _cnvModel(self):
        """
        Convert GPR model for pyJenny
        """
        pass

    def _predict_lib(self, inp, eval_MSE):
        """
        Using the prediction function implemented in the library (sklearn/GPflow) itself
        """
        pass

    def _predict_usr(self, u, eval_MSE):
        """
        The prediction function implemented directly by the user
        This particular implementation applies to sklGPR and gpfGPR
        """
        kern, kndg, mean, mshp = self._getCRFun(jac=False)

        Ksu = kern(self.X, u)
        yp  = Ksu.T.dot(self.gamma) + mean(u)

        if eval_MSE:
            e1 = kndg(u)
            rt = sp.linalg.solve_triangular(self.C, Ksu, lower=True)
            e2 = self._diag(rt)
            tt = self.Ft.T.dot(rt) - mshp(u).T
            gt = sp.linalg.solve_triangular(self.G, tt, lower=True)
            e3 = self._diag(gt)
            es = e1 - e2 + e3
            ss = es.dot(self.sigma2.reshape(1,-1))
            return yp, ss
        return yp

    def _predict_wj(self, u, eval_MSE):
        """
        Prediction at a single point with Jacobian
        """
        kern, kndg, mean, mshp = self._getCRFun(jac=True)

        Ksu, KsuJ = kern(self.X, u)
        m, mJ = mean(u)
        yp  = Ksu.T.dot(self.gamma) + m
        ypJ = self.gamma.T.dot(KsuJ) + mJ

        if eval_MSE:
            e1, e1J = kndg(u)
            h, hJ = mshp(u)
            rt = sp.linalg.solve_triangular(self.C, Ksu, lower=True)
            e2 = self._diag(rt)
            tt = self.Ft.T.dot(rt) - h.reshape(-1,1)
            gt = sp.linalg.solve_triangular(self.G, tt, lower=True)
            e3 = self._diag(gt)
            es = e1 - e2 + e3
            sg = self.sigma2.reshape(1,-1)
            ss = es.dot(sg)

            gtJ = sp.linalg.solve_triangular(self.G.T, gt)
            vkJ = sp.linalg.solve_triangular(self.C.T, self.Ft.dot(gtJ) - rt)
            esJ = e1J + 2 * (vkJ.T.dot(KsuJ) - gtJ.T.dot(hJ))

            ssJ = sg.T.dot(esJ)
            return [yp, ypJ], [ss, ssJ]
        return [yp, ypJ]

    # -----------------------------------------
    # Auxiliary utilities
    # -----------------------------------------
    def _ndData(self, Xdim, Ydim):
        """Nondimensionalize data."""
        mx = np.amax(Xdim, axis=0)
        mn = np.amin(Xdim, axis=0)
        self.X_mean = 0.5 * (mx + mn)
        self.X_std  = mx - mn
        self.nfea = len(self.X_mean)

        mx = np.amax(Ydim, axis=0)
        mn = np.amin(Ydim, axis=0)
        self.y_mean = np.atleast_1d(0.5 * (mx + mn))
        self.y_std  = np.atleast_1d(mx - mn)
        self.nout = len(self.y_mean)

        self.X_ = (Xdim-self.X_mean)/self.X_std  # x-data for training
        Y = Ydim.reshape(-1,self.nout)
        self.Y_ = (Y-self.y_mean)/self.y_std  # y-data for training

        self.ydim_min = np.min(Ydim)

    def _dmData(self):
        """Dimensionalize data."""
        return self.X_std*self.X_ + self.X_mean, self.y_std*self.Y_ + self.y_mean

    def _dmOut(self, out):
        """Dimensionalize output."""
        return self.y_std*out + self.y_mean

    def _dmJac(self, jac):
        """Dimensionalize jacobian."""
        return self.y_std.reshape(-1,1) * jac / self.X_std

    def _dmMSE(self, mse, jac=None):
        """
        Dimensionalize MSE.
        Note that the prediction provides sigma-SQUARED, but the output is the std.
        So both the inputs should go through the sqrt function.
        """
        _m = mse >= 0
        ss = np.zeros_like(mse)
        ss[_m] = np.sqrt(mse[_m])
        if not np.all(_m):
            print("Warning: invalid variance={0}".format(mse[~_m]))

        if jac is not None:
            # Jacobian of sigma^2 w.r.t. input
            _w = np.zeros_like(ss)
            _w[_m] = 0.5 / ss[_m]
            _j = _w.reshape(-1,1) * jac
            return [self.y_std * ss, self._dmJac(_j)]
        return self.y_std * ss

    def _procInp(self, x):
        return (x.reshape(-1, self.nfea) - self.X_mean) / self.X_std

    def _diag(self, m):  # pylint: disable=no-self-use
        return np.sum(m*m, axis=0).reshape(-1,1)

    def _getCRFun(self, jac=False):
        self._corr.setCoef(self.t0)
        self._regr.setCoef(self.beta)
        kern, kndg = self._corr.getFunc(jac=jac)
        mean, mshp = self._regr.getFunc(jac=jac)
        return kern, kndg, mean, mshp

    def _printParam(self):
        self.printMsg("theta:  {0}".format(self.theta_), 2)
        self.printMsg("sigma2: {0}".format(self.sigma2), 2)
        if getattr(self, 'kvar', None) is not None:
            self.printMsg("kvar:   {0}".format(self.kvar), 2)
            self.printMsg("lvar:   {0}".format(self.lvar), 2)
        if getattr(self, 'jitter_level', None) is not None:
            self.printMsg("jitter: {0}".format(self.jitter_level), 2)

# ======================================================================
#         The sklearn-based GPR model
# ======================================================================
class sklGPR(GPR):
    """
    The GPR model based on sklearn implementation.
    The sklearn.gaussian_process.GaussianProcess model is deprecated since version 0.18.
    However, the successor GaussianProcessRegressor does not support explicit mean function.
    Therefore, we stick with the old implementation, which is stored in extern.gaussian_process.
    """
    def __init__(self, **kwargs):
        super(sklGPR, self).__init__(**kwargs)
        self.gpType = 'sklgpr'

    # -----------------------------------------
    # Interface
    # -----------------------------------------
    def fit(self, Xdim, Ydim):
        self._ndData(Xdim, Ydim)
        self._procTheta()

        self.printMsg('', 0)
        self.printMsg("skl: Training GPR", 0)
        if self.noiseLvl is not None:
            nugget = self.noiseLvl
        else:
            nugget = JITTER
        t1 = time.time()
        self._obj = gaussian_process.GaussianProcess(theta0       = self.theta0,
                                                     thetaL       = self.thetaL,
                                                     thetaU       = self.thetaU,
                                                     regr         = self.regr,
                                                     corr         = self.corr,
                                                     nugget       = nugget,
                                                     verbose      = self.verbose,
                                                     normalize    = False,
                                                     random_start = 1)
        try:
            self._obj.fit(self.X_, self.Y_)
            _flg = True
        except ValueError:
            self.printMsg("Failed to optimize for new parameters", -1)
            _flg = False
        t2 = time.time()

        if self._ifCnv:
            if _flg:
                self.printMsg("Converting model for pyJenny", 1)
                self._cnvModel()
            self._printParam()

            _, er = self.score(Xdim, Ydim)
            self.printMsg("err: ({0:f}s, {1:f})".format(t2-t1, np.mean(1-er)), 1)
        else:
            self.printMsg("cost: ({0:f}s)".format(t2-t1), 1)

    # -----------------------------------------
    # Core functions
    # -----------------------------------------
    def _cnvModel(self):
        ents = ['theta_', 'beta', 'gamma', 'C', 'Ft', 'G', 'sigma2']
        for ee in ents:
            setattr(self, ee, getattr(self._obj, ee))
        self.X  = np.array(self.X_)  # x-data for prediction
        if self.corr == 'squared_exponential':
            self.t0 = np.sqrt(2.0*self.theta_)
        else:
            self.t0 = np.copy(self.theta_)

    def _predict_lib(self, inp, eval_MSE):
        return self._obj.predict(inp, eval_MSE)

    # -----------------------------------------
    # Auxiliary utilities
    # -----------------------------------------
    def _procTheta(self):
        if isinstance(self.theta[0], float):
            one = np.ones((self.nfea,))
            self.theta0 = self.theta[0] * one
            self.thetaL = self.theta[1] * one
            self.thetaU = self.theta[2] * one
        else:
            self.theta0 = self.theta[0]
            self.thetaL = self.theta[1]
            self.thetaU = self.theta[2]

# ======================================================================
#         The GPflow-based GPR model
# ======================================================================
class gpfGPR(GPR):
    """
    The GPR model based on vanilla implementation from GPflow.
    """
    def __init__(self, **kwargs):
        super(gpfGPR, self).__init__(**kwargs)
        self.gpType = 'gpfgpr'

    # -----------------------------------------
    # Interface
    # -----------------------------------------
    def fit(self, Xdim, Ydim):
        self._ndData(Xdim, Ydim)
        self._setCRFun()

        self.printMsg('', 0)
        self.printMsg("gpf: Training GPR", 0)
        t1 = time.time()
        self._obj = GPflow.gpr.GPR(self.X_, self.Y_, self.kernFun,
                                   mean_function=self.meanFun)
        if self.noiseLvl is not None:
            self._obj.likelihood.variance = self.noiseLvl
            self._obj.likelihood.variance.fixed = True

        custom_config = GPflow.settings.get_settings()
        custom_config.verbosity.optimisation_verb = self.verbose
        try:
            with GPflow.settings.temp_settings(custom_config):
                self._obj.optimize(method  = self.method,
                                   maxiter = self.maxIter,
                                   maxfun  = 2*self.maxIter)
            _flg = True
        except ValueError:
            self.printMsg("Failed to optimize for new parameters", -1)
            _flg = False
        t2 = time.time()

        if self._ifCnv:
            if _flg:
                self.printMsg("Converting model for pyJenny", 1)
                self._cnvModel()
            self._printParam()

            _, er = self.score(Xdim, Ydim)
            self.printMsg("err: ({0:f}s, {1:f})".format(t2-t1, np.mean(1-er)), 1)
        else:
            self.printMsg("cost: ({0:f}s)".format(t2-t1), 1)

    # -----------------------------------------
    # Core functions
    # -----------------------------------------
    def _cnvModel(self):
        self._cnvModel_cr()
        kern, _, mean, mshp = self._getCRFun(jac=False)

        self.X = np.array(self.X_)  # x-data for prediction
        R = kern(self.X_, self.X_) + self.lvar/self.kvar * np.eye(len(self.X_))
        err = self.Y_ - mean(self.X_)
        self.C = np.linalg.cholesky(R)
        rho = sp.linalg.solve_triangular(self.C, err, lower=True)
        self.gamma = sp.linalg.solve_triangular(self.C.T, rho)
        self.Ft = sp.linalg.solve_triangular(self.C, mshp(self.X_), lower=True)
        _, self.G = sp.linalg.qr(self.Ft, mode='economic')
        sig = self._diag(rho)/len(self.X_)
        self.sigma2 = sig.reshape(-1)
        # The variance used in GPflow
        # In this class, kvar is fixed to be 1.0
        # self.sigma2 = self.kvar

    def _predict_lib(self, inp, eval_MSE):
        y, s = self._obj.predict_f(inp)
        if eval_MSE:
            # return self._obj.predict_y(u)
            return y, s
        return y

    def _cnvModel_cr(self):
        self.lvar = self._obj.likelihood.variance.value

        if self.corr == 'squared_exponential':
            self.t0 = 1.0/self._obj.kern.lengthscales.value
            self.theta_ = 0.5 * self.t0 * self.t0
            self.kvar = self._obj.kern.variance.value

        if self.regr == 'constant':
            self.beta = np.array(self._obj.mean_function.c.value)
        elif self.regr == 'linear':
            b = self._obj.mean_function.b.value
            A = self._obj.mean_function.A.value
            self.beta = np.vstack((b.reshape(1,-1), A.reshape(self.nfea,-1)))

    # -----------------------------------------
    # Auxiliary utilities
    # -----------------------------------------
    def _setCRFun(self):
        if self.corr == "squared_exponential":
            self.kernFun = GPflow.kernels.RBF(self.nfea, ARD=True)
        elif self.corr == "matern12":
            self.kernFun = GPflow.kernels.Matern12(self.nfea, ARD=True)
        elif self.corr == "matern32":
            self.kernFun = GPflow.kernels.Matern32(self.nfea, ARD=True)
        elif self.corr == "matern52":
            self.kernFun = GPflow.kernels.Matern52(self.nfea, ARD=True)
        elif self.corr == "linear":
            self.kernFun = GPflow.kernels.Linear(self.nfea, ARD=True)
        else:
            raise NotImplementedError("Correlation type {0} not implemented!"
                                      "".format(self.corr))
        # The process variance will be estimated later in _cnvModel
        # Here the *normalized* kernel function is desired
        self.kernFun.variance = 1.0
        self.kernFun.variance.fixed = True

        if self.regr == "constant":
            self.meanFun = GPflow.mean_functions.Constant(c=np.zeros((self.nout,)))
        elif self.regr == "linear":
            self.meanFun = GPflow.mean_functions.Linear(A=np.zeros((self.nfea, self.nout)),
                                                        b=np.zeros((self.nout,)))
        else:
            raise NotImplementedError("Regression type {0} not implemented!"
                                      "".format(self.regr))

# ======================================================================
#         The GPflow-based GPRFITC model
# ======================================================================
class gpfGPRFITC(gpfGPR):
    """
    The GPR model based on FITC implementation from GPflow.
    """
    def __init__(self, **kwargs):
        super(gpfGPRFITC, self).__init__(**kwargs)
        self.gpType = 'gpffitc'

    # -----------------------------------------
    # Interface
    # -----------------------------------------
    def fit(self, Xdim, Ydim):
        self._ndData(Xdim, Ydim)
        # if self.nout > 1:
        #     raise NotImplementedError("Output dim {0} > 1 not implemented!".format(self.nout))
        self._setCRFun()

        self.printMsg('', 0)
        self.printMsg("gpf: Training GPRFITC", 0)
        t1 = time.time()
        self._genPseudoInp()
        self._obj = GPflow.sgpr.GPRFITC(self.X_, self.Y_,
                                        self.kernFun, self.Z,
                                        mean_function=self.meanFun)
        custom_config = GPflow.settings.get_settings()
        custom_config.verbosity.optimisation_verb = self.verbose
        if self.noiseLvl is not None:
            custom_config.numerics.jitter_level = self.noiseLvl
            self.jitter_level = self.noiseLvl
            self._obj.likelihood.variance = self.noiseLvl
            self._obj.likelihood.variance.fixed = True
        else:
            self.jitter_level = custom_config.numerics.jitter_level
        try:
            with GPflow.settings.temp_settings(custom_config):
                self._obj.optimize(method  = self.method,
                                   maxiter = self.maxIter,
                                   maxfun  = 2*self.maxIter)
            _flg = True
        except ValueError:
            self.printMsg("Failed to optimize for new parameters", -1)
            _flg = False
        t2 = time.time()

        if self._ifCnv:
            if _flg:
                self.printMsg("Converting model for pyJenny", 1)
                self._cnvModel()
            self._printParam()

            _, er = self.score(Xdim, Ydim)
            self.printMsg("err: ({0:f}s, {1:f})".format(t2-t1, np.mean(1-er)), 1)
        else:
            self.printMsg("cost: ({0:f}s)".format(t2-t1), 1)

    # -----------------------------------------
    # Core functions
    # -----------------------------------------
    def _predict_usr(self, u, eval_MSE):
        kern, kndg, mean, mshp = self._getCRFun(jac=False)

        Ksu = kern(self.X, u)
        yp  = Ksu.T.dot(self.gamma) + mean(u)

        if eval_MSE:
            e1 = kndg(u) #+ self.lvar/self.kvar
            rt = sp.linalg.solve_triangular(self.C, Ksu, lower=True)
            e2 = self._diag(rt)
            # GPR formulation
            tt = self.Ft.T.dot(rt) - mshp(u).T
            gt = sp.linalg.solve_triangular(self.G, tt, lower=True)
            e3 = self._diag(gt)
            # GPRFITC formulation
            # gt = sp.linalg.solve_triangular(self.L, rt, lower=True)
            # e3 = self._diag(gt)
            es = e1 - e2 + e3
            ss = es.dot(self.sigma2.reshape(1,-1))
            return yp, ss
        return yp

    def _cnvModel(self):
        self._cnvModel_cr()
        kern, kndg, mean, mshp = self._getCRFun(jac=False)

        self.X = np.array(self.Z)  # x-data for prediction
        err = self.Y_ - mean(self.X_)
        Kdiag = kndg(self.X_)
        Kuf = kern(self.Z, self.X_)
        R   = kern(self.Z, self.Z) + np.eye(self.numPseudo) * self.jitter_level/self.kvar

        self.C = np.linalg.cholesky(R)
        V = sp.linalg.solve_triangular(self.C, Kuf, lower=True)

        diagQff = self._diag(V)
        nu = Kdiag - diagQff + self.lvar/self.kvar
        B  = np.eye(self.numPseudo) + np.dot(V/nu.T, V.T)
        self.L = np.linalg.cholesky(B)

        t1 = err / nu
        t2 = V.dot(t1)
        t3 = sp.linalg.solve_triangular(self.L, t2, lower=True)
        rho = sp.linalg.solve_triangular(self.L.T, t3, lower=False)
        self.gamma = sp.linalg.solve_triangular(self.C.T, rho, lower=False)
        sig = self._diag(rho)/len(self.Z)
        self.sigma2 = sig.reshape(-1)

        # For GPR formulation
        self.Ft = sp.linalg.solve_triangular(self.C, mshp(self.Z), lower=True)
        _, self.G = sp.linalg.qr(self.Ft, mode='economic')
        # For GPRFITC formulation
        self.Luui = sp.linalg.solve_triangular(self.C, np.eye(self.numPseudo), lower=True)
        self.LLui = sp.linalg.solve_triangular(self.L, self.Luui, lower=True)

    # -----------------------------------------
    # Auxiliary utilities
    # -----------------------------------------
    def _genPseudoInp(self):
        if self.Z is None:
            smpl = Sampling()
            self.Z = smpl(lhsScale = [[-0.5, 0.5]]*self.nfea,
                          numDim   = self.nfea,
                          numSmp   = self.numPseudo,
                          odrSchm  = 'simple')
        else:
            self.Z = (self.Z-self.X_mean)/self.X_std
            self.numPseudo, _ = self.Z.shape

# ======================================================================
#         Classes for correlation functions
# ======================================================================
class Corr(object):
    """Base class for correlation functions"""
    def setCoef(self, t0):
        """
        Set inverse length scale.
        For sklGPR, t0 is converted from theta. The relations are function-dependent.
        For gpfGPR, t0 is given by its kernel.
        """
        self.t0 = np.copy(t0)
        self.t2 = self.t0**2

    def getFunc(self, jac=False):
        """
        Get two functions related to correlation, kern and kndg.
        kern: correlation between two sets of data points.
        kndg: self-correlation.
        """
        pass

class CorrSqre(Corr):
    """Squared exponential."""
    def getFunc(self, jac=False):
        if jac:
            def _kern(x1, x2, _t=self.t2):
                dx = x1-x2
                tx = _t * dx
                d2 = np.sum(dx*tx, axis=1).reshape(-1,1)
                cr = np.exp(-0.5 * d2)
                return cr, cr * tx
            def _kndg(x):
                _n = x.shape[0]
                return np.ones((_n,1)), np.zeros((_n,1))
        else:
            def _kern(x1, x2, _t=self.t0):
                X1  = x1*_t
                X2  = x2*_t
                X1s = np.sum(X1*X1, axis=1).reshape(-1,1)
                X2s = np.sum(X2*X2, axis=1).reshape(1,-1)
                dst = X1s + X2s - 2.0 * X1.dot(X2.T)
                return np.exp(-0.5 * dst)
            def _kndg(x):
                return np.ones((x.shape[0],1))
        return _kern, _kndg

class CorrMt12(Corr):
    """Matern12."""
    def getFunc(self, jac=False):
        if jac:
            def _kern(x1, x2, _t=self.t2):
                dx = x1-x2
                tx = _t * dx
                dt = np.sqrt(np.sum(dx*tx, axis=1).reshape(-1,1))
                cr = np.exp(-dt)
                return cr, (cr/dt) * tx
            def _kndg(x):
                _n = x.shape[0]
                return np.ones((_n,1)), np.zeros((_n,1))
        else:
            def _kern(x1, x2, _t=self.t0):
                X1  = x1*_t
                X2  = x2*_t
                X1s = np.sum(X1*X1, axis=1).reshape(-1,1)
                X2s = np.sum(X2*X2, axis=1).reshape(1,-1)
                dst = X1s + X2s - 2.0 * X1.dot(X2.T)
                dst[dst < 0.0] = 0.0
                rl  = np.sqrt(dst)
                return np.exp(-rl)
            def _kndg(x):
                return np.ones((x.shape[0],1))
        return _kern, _kndg

class CorrMt32(Corr):
    """Matern32."""
    def getFunc(self, jac=False):
        if jac:
            def _kern(x1, x2, _t=self.t2):
                dx = x1-x2
                tx = _t * dx
                dt = np.sqrt(3) * np.sqrt(np.sum(dx*tx, axis=1).reshape(-1,1))
                cr = np.exp(-dt)
                return (1.0+dt) * cr, 3.0*cr * tx
            def _kndg(x):
                _n = x.shape[0]
                return np.ones((_n,1)), np.zeros((_n,1))
        else:
            def _kern(x1, x2, _t=self.t0):
                X1  = x1*_t
                X2  = x2*_t
                X1s = np.sum(X1*X1, axis=1).reshape(-1,1)
                X2s = np.sum(X2*X2, axis=1).reshape(1,-1)
                dst = X1s + X2s - 2.0 * X1.dot(X2.T)
                dst[dst < 0.0] = 0.0
                rl  = np.sqrt(3) * np.sqrt(dst)
                return (1.0 + rl) * np.exp(-rl)
            def _kndg(x):
                return np.ones((x.shape[0],1))
        return _kern, _kndg

class CorrMt52(Corr):
    """Matern52."""
    def getFunc(self, jac=False):
        if jac:
            def _kern(x1, x2, _t=self.t2):
                dx = x1-x2
                tx = _t * dx
                dt = np.sqrt(5) * np.sqrt(np.sum(dx*tx, axis=1).reshape(-1,1))
                cr = np.exp(-dt)
                return (1.0+dt+dt*dt/3.0) * cr, 5.0/3.0*(1.0+dt)*cr * tx
            def _kndg(x):
                _n = x.shape[0]
                return np.ones((_n,1)), np.zeros((_n,1))
        else:
            def _kern(x1, x2, _t=self.t0):
                X1  = x1*_t
                X2  = x2*_t
                X1s = np.sum(X1*X1, axis=1).reshape(-1,1)
                X2s = np.sum(X2*X2, axis=1).reshape(1,-1)
                dst = X1s + X2s - 2.0 * X1.dot(X2.T)
                dst[dst < 0.0] = 0.0
                rl  = np.sqrt(5) * np.sqrt(dst)
                return (1.0 + rl + rl*rl/3.0) * np.exp(-rl)
            def _kndg(x):
                return np.ones((x.shape[0],1))
        return _kern, _kndg

# ======================================================================
#         Classes for regression functions
# ======================================================================
class Regr(object):
    """Base class for regression functions"""
    def setCoef(self, beta):
        """Set regression coefficients."""
        self.beta = np.copy(beta)

    def getFunc(self, jac=False):
        """
        Get two functions related to regression, mean and mshp.
        mean: regression function itself.
        mshp: shape functions in mean.
        """
        pass

class RegrCnst(Regr):
    """Constant."""
    def getFunc(self, jac=False):
        if jac:
            def _mean(x, _b=self.beta):  # pylint: disable=unused-argument
                _bb = _b.reshape(1,-1)
                return _bb, np.zeros_like(_bb)
            def _mshp(x):  # pylint: disable=unused-argument
                return np.ones((1,1)), np.zeros((1,1))
        else:
            def _mean(x, _b=self.beta):
                return np.ones((x.shape[0],1)).dot(_b.reshape(1,-1))
            def _mshp(x):
                return np.ones((x.shape[0],1))
        return _mean, _mshp

class RegrLinr(Regr):
    """Linear."""
    def getFunc(self, jac=False):
        if jac:
            def _mean(x, _b=self.beta):
                return x.dot(_b[1:]) + _b[0], _b[1:].T
            def _mshp(x):
                _s = np.hstack(([1], x.reshape(-1)))
                _j = np.eye(len(_s))
                return _s, _j[:,1:]
        else:
            def _mean(x, _b=self.beta):
                return x.dot(_b[1:]) + _b[0]
            def _mshp(x):
                return np.hstack((np.ones((x.shape[0],1)), x))
        return _mean, _mshp
