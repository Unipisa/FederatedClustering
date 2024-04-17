#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr. 15 9:37 a.m. 2024

@author: AI&RD Research Group, Department of Information Engineering, University of Pisa
"""
from setuptools import setup

setup(name="FedFuzzClus",
      version="0.1",
      description="Federated c-means and Fuzzy c-means Clustering Algorithms for Horizontally and Vertically Partitioned Data library.",
      author="AI&RD Research Group",
      author_email="info@ai.dii.unipi.it",
      packages=["FedFuzzClus"],
      install_requires=['numpy==1.26.4', 'numba==0.59.1', 'matplotlib==3.8.4', 'pandas==2.2.2', 'scikit-learn==1.4.2'],
      include_package_data=True,
      long_description=open('README.md').read()
      )
