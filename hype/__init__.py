#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from . import manifolds
from . import energy_function
import argparse
import logging

MANIFOLDS = {
    'lorentz': manifolds.LorentzManifold,
    'poincare': manifolds.PoincareManifold,
    'euclidean': manifolds.EuclideanManifold,
}

MODELS = {
    'distance': energy_function.DistanceEnergyFunction,
    'entailment_cones': energy_function.EntailmentConeEnergyFunction,
}


def build_model(opt, N):
    if isinstance(opt, argparse.Namespace):
        opt = vars(opt)

    #log = logging.getLogger('poincare')
    #log.info('\n\n ------------------------------ \n Poincare model initialised. \n See hype/__init__.py comments \n for details of training loss. \n ------------------------------ \n\n')

    # train.sh specifies a model (default=distance) and manifold (poincare)
    # these are read by parser in embed.py and stored in opt dictionary
    # used here to select from above MANIFOLDS and MODELS dicts to create model
    # also size=N (num vecs) whilst dim, sparse and margin stored in opt also

    # Note this selection from the MODELS dict means we return a
    # DistanceEnergyFunction (we are not doing Entailment Cones!)
    # Importantly DistanceEnergyFunction defines a cross-entropy loss fn for
    # training. Comparing https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    # with 3.6 thesis shows this is equivalent to the soft ranking loss we need

    K = 0.1 if opt['model'] == 'entailment_cones' else None
    manifold = MANIFOLDS[opt['manifold']](K=K)
    return MODELS[opt['model']](
        manifold,
        dim=opt['dim'],
        size=N,
        sparse=opt['sparse'],
        margin=opt['margin']
    )
