# LOSP_Plus
These codes are for our paper "Krylov Subspace Approximation for Local Community Detection in Large Networks"
## Requirements
Before compiling codes, the following software should be installed in your system.
- Matlab
- gcc (for Linux and Mac) or Microsoft Visual Studio (for Windows)
## Datasets Information
- SNAP datasets (available at http://snap.stanford.edu/data/)
- Biology datasets (available at http://cb.csail.mit.edu/cb/mna/isobase/) 
- LFR benchmark graphs (available at http://sites.google.com/site/santofortunato/inthepress2/)
### Example dataset
- Amazon dataset (available at http://snap.stanford.edu/data/com-Amazon.html)
- nodes: 334863, edges: 925872 
- nodes are products, edges are co-purchase relationships
- top 5000 communities with ground truth size >= 3
## How to run LOSP_Plus algorithm
```
$ cd LOSP_Plus_codes
$ matlab 
$ mex -largeArrayDims GetLocalCond.c   % compile the mex file 
$ LOSP_Plus(WalkMode,d,k,alpha,TruncateMode,beta) 
```
### Command Options for LOSP_Plus algorithm:

WalkMode: 1: standard, 2: light lazy, 3: lazy, 4: personalized (default: 2)

d: dimension of local spectral subspace (default: 2)

k: number of random walk steps (default: 3)

alpha: a parameter controls random walk diffusion (default: 1)

TruncateMode: 1: truncation by truth size, 2: truncation by local minimal conductance (default: 2)

beta: a parameter controls local minimal conductance (default: 1.02)
## How to run baseline algorithms
### run LEMON algorithm
```
$ cd baseline_codes/LEMON
$ matlab 
$ LEMON
```
### run PGDC-d algorithm
```
$ cd baseline_codes/PGDc-d
$ matlab
$ PGDC_d
```
### run HK algorithm
```
$ cd baseline_codes/HK
$ matlab 
$ mex -largeArrayDims hkgrow_mex.cpp   % compile the mex file 
$ HK
```
### run PR algorithm
```
$ cd baseline_codes/PR
$ matlab 
$ mex -largeArrayDims pprgrow_mex.cc   % compile the mex file 
$ PR
```
## More comparison
### run GLOSP algorithm for evaluating the effectiveness of Krylov subspace approximation
```
$ cd LOSP_Plus_codes
$ matlab 
$ GLOSP_Plus   % GLOSP algorithm using eigenspace rather than Krylov subspace
```
### run HKL and PRL for evaluating the effectiveness of local minimal conductance truncation
```
$ cd LOSP_Plus_codes
$ matlab
$ mex -largeArrayDims hkvec_mex.cpp   % compile mex file
$ mex -largeArrayDims pprvec_mex.cc   % compile mex file
$ HK_local    % HKL algorithm based on local minimal conductance truncation
$ PR_local    % PRL algorithm based on local minimal conductance truncation
```
## Announcements
### Licence
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://fsf.org/.
### Notification
Please email to panshi@hust.edu.cn or setup an issue if you have any problems or find any bugs.
### Acknowledgement
In the program, we incorporate some open source codes as baseline algorithms from the following websites:
- [LEMON](https://github.com/yixuanli/lemon)
- [PGDC-d](http://cs.ru.nl/~tvanlaarhoven/conductance2016/)
- [HK](https://github.com/kkloste/hkgrow)
- [PR](https://www.cs.purdue.edu/homes/dgleich/codes/neighborhoods/)
