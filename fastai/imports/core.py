"""
  csv                                     	CSV file reading and writing            
  gc                                      	Garbage collector interface             
  gzip                                    	Support for gzip files                  
  os                                      	Miscellaneous operating system interfaces
  pickle                                  	Python object serialization             
  shutil                                  	High level file operations              
  sys                                     	System-specific parameters and functions
  warnings, warn                          	Warning control                         
  yaml                                    	YAML parser and emitter                 
  io, BufferedWriter, BytesIO             	Core tools for working with streams     
  subprocess                              	Subprocess management                   
  math                                    	Mathematical functions                  
  plt(matplotlib.pyplot)                  	MATLAB-like plotting framework          
  np (numpy) , array, cos, exp, log, sin, tan, tanh	Multi-dimensional arrays, mathematical functions
  pd (pandas), Series, DataFrame          	Data structures and tools for data analysis
  random                                  	Generate pseudo-random numbers          
  scipy.stats                             	Statistical functions                   
  scipy.special                           	Special functions                       
  abstractmethod, abstractproperty        	Abstract base classes                   
  collections, Counter, defaultdict,  namedtuple, OrderedDict	Container datatypes                     
  abc (collections.abc), Iterable         	Abstract base classes for containers    
  hashlib                                 	Secure hashes and message digests       
  itertools                               	Functions creating iterators for efficient looping
  json                                    	JSON encoder and decoder                
  operator, attrgetter, itemgetter        	Standard operators as functions         
  pathlib, Path                           	Object-oriented filesystem paths        
  mimetypes                               	Map filenames to MIME types             
  inspect                                 	Inspect live objects                    
  typing, Any, AnyStr, Callable, Collection, Dict, Hashable, Iterator, List, Mapping, NewType, Optional, Sequence, Tuple, TypeVar, Union	Support for type hints                  
  functools, partial, reduce              	Higher-order functions and operations on callable objects
  importlib                               	The implementatin of import             
  weakref                                 	Weak references                         
  html                                    	HyperText Markup Language support       
  re                                      	Regular expression operations           
  requests                                	HTTP for Humans™                        
  tarfile                                 	Read and write tar archive files        
  numbers, Number                         	Numeric abstract base classes           
  tempfile                                	Generate temporary files and directories
  concurrent, ProcessPoolExecutor, ThreadPoolExecutor	Launch parallel tasks                   
  copy, deepcopy                          	Shallow and deep copy operation         
  dataclass, field, InitVar               	Data Classes                            
  Enum, IntEnum                           	Support for enumerations                
  set_trace                               	The Python debugger                     
  patches (matplotlib.patches), Patch     	?                                       
  patheffects (matplotlib.patheffects)    	?                                       
  contextmanager                          	Utilities for with-statement contexts   
  MasterBar, master_bar, ProgressBar, progress_bar	Simple and flexible progress bar for Jupyter Notebook and console
  pkg_resources                           	Package discovery and resource access   
  SimpleNamespace                         	Dynamic type creation and names for built-in types
  torch, as_tensor, ByteTensor, DoubleTensor, FloatTensor, HalfTensor, LongTensor, ShortTensor, Tensor	Tensor computation and deep learning    
  nn (torch.nn), weight_norm, spectral_norm	Neural networks with PyTorch            
  F(torch.nn.functional)                  	PyTorch functional interface            
  optim (torch.optim)                     	Optimization algorithms in PyTorch      
  BatchSampler, DataLoader, Dataset, Sampler, TensorDataset	PyTorch data utils                      
"""
import csv, gc, gzip, os, pickle, shutil, sys, warnings, yaml, io, subprocess
import math, matplotlib.pyplot as plt, numpy as np, pandas as pd, random
import scipy.stats, scipy.special
import abc, collections, hashlib, itertools, json, operator, pathlib
import mimetypes, inspect, typing, functools, importlib, weakref
import html, re, requests, tarfile, numbers, tempfile

from abc import abstractmethod, abstractproperty
from collections import abc,  Counter, defaultdict, Iterable, namedtuple, OrderedDict
import concurrent
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import copy, deepcopy
from dataclasses import dataclass, field, InitVar
from enum import Enum, IntEnum
from functools import partial, reduce
from pdb import set_trace
from matplotlib import patches, patheffects
from numpy import array, cos, exp, log, sin, tan, tanh
from operator import attrgetter, itemgetter
from pathlib import Path
from warnings import warn
from contextlib import contextmanager
from fastprogress.fastprogress import MasterBar, ProgressBar
from matplotlib.patches import Patch
from pandas import Series, DataFrame
from io import BufferedWriter, BytesIO

import pkg_resources
pkg_resources.require("fastprogress>=0.1.19")
from fastprogress.fastprogress import master_bar, progress_bar

#for type annotations
from numbers import Number
from typing import Any, AnyStr, Callable, Collection, Dict, Hashable, Iterator, List, Mapping, NewType, Optional
from typing import Sequence, Tuple, TypeVar, Union
from types import SimpleNamespace

def try_import(module):
    "Try to import `module`. Returns module's object on success, None on failure"
    try: return importlib.import_module(module)
    except: return None

def have_min_pkg_version(package, version):
    "Check whether we have at least `version` of `package`. Returns True on success, False otherwise."
    try:
        pkg_resources.require(f"{package}>={version}")
        return True
    except:
        return False
