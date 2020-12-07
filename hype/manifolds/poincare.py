#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
from torch.autograd import Function
from .euclidean import EuclideanManifold
import numpy as np
import time

class PoincareManifold(EuclideanManifold):
    def __init__(self, eps=1e-5, K=None, **kwargs):
        self.eps = eps
        super(PoincareManifold, self).__init__(max_norm=1 - eps)
        self.K = K
        if K is not None:
            self.inner_radius = 2 * K / (1 + np.sqrt(1 + 4 * K * self.K))

    def distance(self, u, v):
        return Distance.apply(u, v, self.eps)

    def half_aperture(self, u):
        eps = self.eps
        sqnu = u.pow(2).sum(dim=-1)
        sqnu.clamp_(min=0, max=1 - eps)
        return th.asin((self.inner_radius * (1 - sqnu) / th.sqrt(sqnu))
            .clamp(min=-1 + eps, max=1 - eps))

    def angle_at_u(self, u, v):
        norm_u = u.norm(2, dim=-1)
        norm_v = v.norm(2, dim=-1)
        dot_prod = (u * v).sum(dim=-1)
        edist = (u - v).norm(2, dim=-1)  # euclidean distance
        num = (dot_prod * (1 + norm_v ** 2) - norm_v ** 2 * (1 + norm_u ** 2))
        denom = (norm_v * edist * (1 + norm_v**2 * norm_u**2 - 2 * dot_prod).sqrt())
        return (num / denom).clamp_(min=-1 + self.eps, max=1 - self.eps).acos()

    def rgrad(self, p, d_p):
        if d_p.is_sparse:
            p_sqnorm = th.sum(
                p[d_p._indices()[0].squeeze()] ** 2, dim=1,
                keepdim=True
            ).expand_as(d_p._values())
            n_vals = d_p._values() * ((1 - p_sqnorm) ** 2) / 4
            n_vals.renorm_(2, 0, 5)
            d_p = th.sparse.DoubleTensor(d_p._indices(), n_vals, d_p.size())
        else:
            p_sqnorm = th.sum(p ** 2, dim=-1, keepdim=True)
            d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
        return d_p

    def acosh(self,x):
        return th.log(x+(x**2-1)**0.5)

    def norm(self, u):
        return th.sqrt(th.sum(th.pow(u, 2), dim=-1))

    def sqnorm(self, u):
        return th.sum(th.pow(u, 2), dim=-1)

    def sparse_norm(self, u):
        i = u._indices()
        v = u._values()
        return th.sparse.FloatTensor(i, th.sqrt(th.sum(th.pow(v,2), dim=-1)), th.Size([u.shape[0]]))

    def sparse_sqnorm(self, u):
        i = u._indices()
        v = u._values()
        return th.sparse.FloatTensor(i, th.sum(th.pow(v,2), dim=-1), th.Size([u.shape[0]]))

    def poincare_inner_product(self, a, b, sparse_flag=False):
        """
        See 2.34 thesis
        """
        if sparse_flag == False:
        # g_numerator = 2 * th.sum( (a-b) ** 2, dim=-1, keepdim=False)
        # g_denominator = (1 - th.sum(a ** 2, dim=-1, keepdim=False)) * (1 - th.sum(b ** 2, dim=-1, keepdim=False))
        # g_angle = th.ones(a.shape[0]) + g_numerator / g_denominator
        # return self.acosh(g_angle)
            numerator = 2 * self.sqnorm(a-b)
            denominator = th.mul( (th.ones_like(numerator) - self.sqnorm(a)), (th.ones_like(numerator) - self.sqnorm(b)) )
            return self.acosh( th.addcdiv(th.ones_like(numerator), numerator, denominator) )
        else:
            i = a._indices()
            v = a._values()
            bv = b[i[:]][0] # get corresponding b values
            diff_vec = th.sparse.FloatTensor(i, v - bv, a.size())
            numerator = 2 * self.sparse_sqnorm(diff_vec)
            # print("SPARSE INNER PRODUCT numerator shape = {}".format(numerator.shape))
            b_sparse_match = th.sparse.FloatTensor(i, bv, a.size())
            # print("b_sparse_match shape = {}".format(b_sparse_match.shape))
            denominator = th.sparse.FloatTensor(i, ( 1 - th.sum(th.pow(v,2), dim=-1) ) * ( 1 - th.sum(th.pow(bv,2), dim=-1) ), th.Size([a.shape[0]]))
            # denominator = th.mul( (th.ones_like(numerator) - self.sparse_sqnorm(a)), (th.ones_like(numerator) - self.sparse_sqnorm(b)) )
            # print("SPARSE INNER PRODUCT denominator shape = {}".format(denominator.shape))
            v_num = numerator._values()
            v_den = numerator._values()
            return th.sparse.FloatTensor(i, self.acosh(1 + v_num / v_den), th.Size([a.shape[0]]))



    def sparse_dense_mul(self, s, d, sc=None):
        i = s._indices()
        v = s._values()
        dv = d[i[:]]  # get values from relevant entries of dense vector
        # print("v shape = {}".format(v.shape))
        # print("dv shape = {}".format(dv.shape))
        if sc is not None:
          return th.sparse.FloatTensor(i, sc * v * dv[0,:], s.size()) # v shape is [sparse] whilst dv shape is [1,sparse]
        else:
          return th.sparse.FloatTensor(i, v * dv[0,:], s.size())

    def dense_div_sparse(self, s, d):
        i = s._indices()
        v = s._values()
        dv = d[i[:]]  # get values from relevant entries of dense vector
        # print("v shape = {}".format(v.shape))
        # print("dv shape = {}".format(dv.shape))
        return th.sparse.FloatTensor(i, dv[0,:] / v, s.size()) # v shape is [sparse] whilst dv shape is [1,sparse]

    def sparsemat_div_sparsevec(self, s_m, s_v):
        i = s_v._indices()
        v = s_v._values()
        M = s_m._values()  # get values from relevant entries of sparse matrix
        # print("M shape = {}".format(M.shape))
        # print("v shape = {}".format(v.shape))
        x = (M.t() / v).t()
        # print(f"x shape = {x.shape}")
        return th.sparse.FloatTensor(i, x, s_m.shape)

    def sparsevec_densemat_mul(self, s, d):
        i = s._indices()
        v = s._values()
        dv = d[i[:],:]  # get values from relevant entries of dense matrix
        # print("v shape = {}".format(v.shape))
        # print("dv shape = {}".format(dv[0,:,:].shape))
        x = (dv[0,:,:].t() * v).t()
        return th.sparse.FloatTensor(i, x, (s.shape[0],d.shape[1])) # v shape is [sparse] whilst dv shape is [1,sparse]

    def sparsevec_sparsemat_mul(self, s, d):
        i = s._indices()
        v_s = s._values()
        v_d = d._values()  # get values from relevant entries of sparse matrix (same entries as sparse vec)
        # print("v shape = {}".format(v.shape))
        # print("dv shape = {}".format(dv[0,:,:].shape))
        x = (v_d.t() * v_s).t()
        return th.sparse.FloatTensor(i, x, (s.shape[0],d.shape[1])) # v shape is [sparse] whilst dv shape is [1,sparse]


    def densevec_sparsemat_mul(sel, d, s):
        i = s._indices()
        v_s = s._values()
        v_d = d[i[:]]  # get values from relevant entries of sparse matrix (same entries as sparse vec)
        x = (v_s.t() * v_d).t()
        return th.sparse.FloatTensor(i, x, s.shape) #  (s.shape[0],d.shape[1])) # v shape is [sparse] whilst dv shape is [1,sparse]

    def sparsevec_sum(self, a, b):
        i = a._indices()
        v_a = a._values()
        v_b = b._values()
        return th.sparse.FloatTensor(i, v_a + v_b, a.shape)

    def sparsevec_densevec_sum(self, s, d):
        i = s._indices()
        v = s._values()
        dv = d[i[:]][0]
        return th.sparse.FloatTensor(i, v + dv, s.shape)

    def sparsevec_sparsevec_mul(self, s1, s2):
        i = s1._indices()
        v1 = s1._values()
        v2 = s2._values()
        return th.sparse.FloatTensor(i, v1 * v2, s1.shape)

    def sparsevec_sparsevec_div(self, s1, s2):
        i = s1._indices()
        v1 = s1._values()
        v2 = s2._values()
        # print("sparse div shape = {}".format(s1.shape))
        return th.sparse.FloatTensor(i, v1 / v2, s1.shape)

    def densevec_sparsevec_mul(self, d, s):
        i = s._indices()
        v = s._values()
        dv = d[i[:]][0]
        return th.sparse.FloatTensor(i, v * dv, s.shape)


    def sparse_cosh_clamp(self, s, minimum, maximum):
        i = s._indices()
        v = s._values()
        v_clamp = th.cosh(th.clamp(v,min=minimum, max=maximum))
        return th.sparse.FloatTensor(i, v_clamp, s.size())

    def sparse_tanh(self, s):
        # tanh bounded so no need to clamp
        i = s._indices()
        v = s._values()
        return th.sparse.FloatTensor(i, th.tanh(v), s.size())

    def sparse_normalised(self, s):
        i = s._indices()
        v = s._values()
        # row-wise norm and clamp to avoid blowup when normalising original v
        v_norm = th.clamp(self.norm(v), min=self.eps)
        v_norm_reciprocal = 1 / v_norm
        # return v_hat
        v_hat = (v.t() * v_norm_reciprocal).t()
        return th.sparse.FloatTensor(i, v_hat, s.size())

    def expm(self, p, d_p, lr=None, out=None, normalize=False):

        """Exponential map for Poincare"""
        if out is None:
            out = p
        # rename variables for consistency with thesis calculation
        theta = out # theta has shape [batch, dim]
        nu = d_p    # nu has shape [batch, dim]
        bsize = theta.shape[0]
        # print(f"theta = {theta}")
        # print(f"nu = {nu}")
        # verbose = True
        # if verbose:
            # print(f"dim = {theta.shape[1]}")
            # print(f"theta shape: {theta.shape}")

        if nu.is_sparse:
            ix, d_val = nu._indices().squeeze(), nu._values()
            # get modulus of nu
            # t0 = time.time()
            mod_nu = self.sparse_norm(nu) # d-1 dims
            print(f"mod_nu CUDA = {mod_nu.is_cuda}")
            # print(f"|nu| ={mod_nu}")
            # t1 = time.time()
            # print("mod_nu calculation time = {}".format(t1-t0))
            # if verbose:
                # print(f"nu shape: {nu.shape}")
                # print(f"|nu| shape: {mod_nu.shape}")

            # print("Is mod_nu sparse? {}".format(mod_nu.is_sparse))
            # get zeta from 3.13 thesis - zeta has shape [batch]
            # zeta = 2.0 / (1.0 - th.sum(theta ** 2, dim=-1, keepdim=False))
            # t2 = time.time()
            # print(f"zeta top = {(2 * th.ones([bsize]))}")
            # print(f"theta sqnorm = {self.sqnorm(theta)}")
            # print(f"zeta bottom = {(th.ones([bsize]) - self.sqnorm(theta))}")
            zeta = th.div( (2 * th.ones([bsize])), (th.ones([bsize]) - self.sqnorm(theta)) )
            # print(f"zeta = {zeta}")
            # t3 = time.time()
            # print("zeta calculation time = {}".format(t3-t2))
            #zeta = 2.0 / (1.0 - th.norm(theta, dim=d-1,p=2) ** 2)
            # if verbose:
                # print(f"zeta shape = {zeta.shape}")

            # compute hyperbolic angle - angle has shape [batch]
            # t4 = time.time()
            if lr is not None:
                # angle = lr * zeta * mod_nu.to_dense()
                # print(mod_nu.shape)
                angle = self.sparse_dense_mul(mod_nu, zeta, lr)
            else:
                # angle = zeta * mod_nu.to_dense()
                angle = self.sparse_dense_mul(mod_nu, zeta)
            # t5 = time.time()
            # print("angle calculation time = {}".format(t5-t4))
            # if verbose:
                # print(f"angle shape = {angle.shape}")
            # print(f"angle = {angle}")

            # calculate cosh(angle) but restrict angle to between -20 and 20 to avoid blowing up
            # t6 = time.time()
            # angle_max = 20 * th.ones([theta.shape[0]])
            # angle_min = -20 * th.ones([theta.shape[0]])
            # cosh = th.cosh(th.min(th.max(angle, angle_max),angle_min))
            cosh = self.sparse_cosh_clamp(angle, minimum=-20, maximum=20)
            # print(f"cosh = {cosh}")
            # t7 = time.time()
            # print("cosh calculation time = {}".format(t7-t6))
            # if verbose:
                # print(f"cosh shape = {cosh.shape}")

            # calculate tanh(angle) - no need to restrict angle since tanh is bounded
            # t_7i = time.time()
            tanh = self.sparse_tanh(angle) # d-1 dims
            # print(f"tanh = {tanh}")
            # t_7ii = time.time()
            # print("tanh calculation time = {}".format(t_7ii-t_7i))
            # if verbose:
                # print(f"tanh shape = {tanh.shape}")

            # get normalised nu vector - threshold denominator to avoid blowup
            # t8 = time.time()
            # print(f"nu shape = {nu.shape}")
            nu_hat = self.sparse_normalised(nu)
            # print(f"nu_hat = {nu_hat}")
            # print(f"nu_hat shape = {nu_hat.shape}")
            #mod_nu_min = 1e-6 * th.ones([mod_nu.shape[0]])
            #bounded_mod_nu = th.max(mod_nu.to_dense(), mod_nu_min) # shape [batch]
            #bounded_mod_nu_resized = bounded_mod_nu.unsqueeze(1).repeat(1,10) # shape [batch, dim] so can divide nu elementwise below
            #normalized_nu = nu.to_dense() / bounded_mod_nu_resized # d dims
            # t9 = time.time()
            # print("normalized_nu calculation time = {}".format(t9-t8))
            # if verbose:
                # print(f"nu_hat shape = {nu_hat.shape}")
                # print(f"bounded_mod_nu shape = {bounded_mod_nu.shape}")
                # print(f"bounded_mod_nu_resized shape = {bounded_mod_nu_resized.shape}")
                # print(f"normalized_nu shape = {normalized_nu.shape}")

            # a = self.poincare_inner_product(theta,normalized_nu)
            # inner product of theta with normalized_nu
            # theta_dot_norm_v = (theta * normalized_nu).sum(axis=d-1)  # d-1 dims                    ##### possibly inconsistent with 2.34 thesis
            # t10 = time.time()
            # numerator_a =  zeta.unsqueeze(1).repeat(1,10) * theta
            numerator_a = (theta.t() * zeta).t()
            # print(f"numerator_a = {numerator_a}")
            # t10i = time.time()
            # print(f"numerator_a shape = {numerator_a.shape}")
            # print(f"numerator_a calculation time = {t10i-t10}")
            # nu_hat_dense = nu_hat.to_dense()
            # t10ii = time.time()
            numerator_b_dotprod = self.poincare_inner_product(nu_hat, theta, sparse_flag=True)
            # t10iia = time.time()
            # print("numerator_b_dotprod calculation time = {}".format(t10iia-t10ii))
            # t10iib = time.time()
            # numerator_b_temp1 = (theta.t() * numerator_b_dotprod).t()
            numerator_b_temp1 = self.sparsevec_densemat_mul(numerator_b_dotprod, theta)
            # t10iic = time.time()
            # print("numerator_b_temp1 calculation time = {}".format(t10iic-t10iib))
            # t10iid = time.time()
            # numerator_b_temp2 = (numerator_b_temp1.t() * zeta).t()
            numerator_b_temp2 = self.densevec_sparsemat_mul(zeta, numerator_b_temp1)
            # t10iie = time.time()
            # print("numerator_b_temp2 calculation time = {}".format(t10iid-t10iie))
            # numerator_b_temp =  zeta.unsqueeze(1).repeat(1,10) * self.poincare_inner_product(theta, nu_hat_dense).unsqueeze(1).repeat(1,10) * theta
            numerator_b = self.sparsevec_sparsemat_mul(tanh, numerator_b_temp2)
            # print(f"numerator_b = {numerator_b}")
            # t10iii = time.time()
            # print(f"numerator_b shape = {numerator_b.shape}")
            # print(f"numerator_b calculation time = {t10iii-t10ii}")
            # t10iv = time.time()
            numerator_c = self.sparsevec_sparsemat_mul(tanh, nu_hat)
            # print(f"numerator_c = {numerator_c}")
            # t10v = time.time()
            # print(f"numerator_c shape = {numerator_c.shape}")
            # print(f"numerator_c calculation time = {t10iv-t10v}")
            # t10vi = time.time()
            numerator = self.sparsevec_sum( self.sparsevec_densevec_sum(numerator_b, numerator_a), numerator_c)
            # print(f"numerator = {numerator}")
            # numerator = numerator_a + numerator_b + numerator_c
            # t10vii = time.time()
            # print(f"numerator sum calculation time = {t10vi-t10vii}")
            # t11 = time.time()
            # print("numerator calculation time = {}".format(t11-t10))
            # if verbose:
                # print(f"numerator shape = {numerator.shape}")

            # t12 = time.time()
            # denominator_a = th.ones([theta.shape[0]]) / cosh
            denominator_a = self.dense_div_sparse(cosh, th.ones([theta.shape[0]]))
            # print(f"denominator_a = {denominator_a}")
            # t12a = time.time()
            # print("denominator_a calculation time = {}".format(t12a-t12))
            # print(f"denominator_a shape = {denominator_a.shape}")
            # t12b = time.time()
            denominator_b = zeta - th.ones([theta.shape[0]])
            # print(f"denominator_b = {denominator_b}")
            # t12c = time.time()
            # print("denominator_b calculation time = {}".format(t12c-t12b))
            # print(f"denominator_b shape = {denominator_b.shape}")
            # t12d = time.time()
            # denominator_c = zeta * self.poincare_inner_product(nu_hat, theta, sparse_flag=True) * tanh
            denominator_c_temp1 = self.poincare_inner_product(nu_hat, theta, sparse_flag=True)
            denominator_c_temp2 = self.sparsevec_sparsevec_mul(tanh, denominator_c_temp1)
            denominator_c = self.densevec_sparsevec_mul(zeta, denominator_c_temp2)
            # print(f"denominator_c = {denominator_c}")
            # t12e = time.time()
            # print("denominator_c calculation time = {}".format(t12e-t12d))
            # print(f"denominator_c shape = {denominator_c.shape}")
            # denominator = denominator_a + denominator_b + denominator_c
            denominator = self.sparsevec_sum( self.sparsevec_densevec_sum(denominator_a, denominator_b), denominator_c)
            # print(f"denominator = {denominator}")
            # t13 = time.time()
            # print("denominator calculation time = {}".format(t13-t12))
            # if verbose:
                # print(f"denominator shape = {denominator.shape}")


            # return numerator / denominator.unsqueeze(1).repeat(1,10)
            # print("numerator shape = {}".format(numerator.shape))
            # print("numerator is sparse: {}".format(numerator.is_sparse))
            # print("denominator shape = {}".format(denominator.shape))
            # print("denominator is sparse: {}".format(denominator.is_sparse))
            result = self.sparsemat_div_sparsevec(numerator, denominator)
            # result = self.sparsevec_sparsevec_div(numerator, denominator)
            #print(f"result = {result}")
            # print(ix)
            # print("p shape = {}".format(p.shape))
            # print("result shape = {}".format(result.shape))
            # print(result)
            # print(result._values())
            p.index_copy_(0, ix, result._values())
            # return self.sparsevec_sparsevec_div(numerator, denominator)
        else:
            return 0

        # numitor = 1 / cosh + (lam_x - 1) + lam_x * x_dot_norm_v * tanh # d-1 dims
        # return numerator / (numitor[:, np.newaxis])
        #return out



class Distance(Function):
    @staticmethod
    # 3.21 thesis
    # N.B. expand alpha * sqrt(gamma^2-1) from thesis and expand z below to identify
    # Also, show brackets in 3.21 equals a
    def grad(x, v, sqnormx, sqnormv, sqdist, eps):
        alpha = (1 - sqnormx)
        beta = (1 - sqnormv)
        z = 1 + 2 * sqdist / (alpha * beta)
        a = ((sqnormv - 2 * th.sum(x * v, dim=-1) + 1) / th.pow(alpha, 2))\
            .unsqueeze(-1).expand_as(x)
        a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
        z = th.sqrt(th.pow(z, 2) - 1)
        z = th.clamp(z * beta, min=eps).unsqueeze(-1)
        # print(f"z shape = {z.shape}")
        return 4 * a / z.expand_as(x)

    @staticmethod
    # 3.20 thesis
    def forward(ctx, u, v, eps):
        squnorm = th.clamp(th.sum(u * u, dim=-1), 0, 1 - eps)
        sqvnorm = th.clamp(th.sum(v * v, dim=-1), 0, 1 - eps)
        sqdist = th.sum(th.pow(u - v, 2), dim=-1)
        ctx.eps = eps
        ctx.save_for_backward(u, v, squnorm, sqvnorm, sqdist)
        x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
        # arcosh
        z = th.sqrt(th.pow(x, 2) - 1)
        return th.log(x + z)

    @staticmethod
    def backward(ctx, g):
        u, v, squnorm, sqvnorm, sqdist = ctx.saved_tensors
        g = g.unsqueeze(-1)
        gu = Distance.grad(u, v, squnorm, sqvnorm, sqdist, ctx.eps)
        gv = Distance.grad(v, u, sqvnorm, squnorm, sqdist, ctx.eps)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv, None
