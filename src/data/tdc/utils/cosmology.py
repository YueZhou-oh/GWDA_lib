import math

import scipy

from .constant import Constant


class Cosmology(object):
    @staticmethod
    def H(zp, w):
        fn = 1.0 / (Constant.H0 * math.sqrt(Constant.Omegam * math.pow(1.0 + zp, 3.0) + Constant.Omegalam * math.pow(1.0 + zp, 3.0 * w)))
        return fn

    @staticmethod
    def DL(zup, w):
        """
        Usage: DL(3,w=0)[0]
        """
        pd = scipy.integrate.quad(Cosmology.H, 0.0, zup, args=(w))[0]
        res = (1.0 + zup) * pd  # in Mpc
        return res * Constant.C_SI * 1.0e-3, pd * Constant.C_SI * 1.0e-3
