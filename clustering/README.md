# density-based and hierarchical_cluster README 

### Requirement

```
$ pip install pandas matplotlib sklearn
```

### Run

Test Pass on Python 2.7

```
$ python hierarchical_cluster.py
$ python density-based.py
$ python3 kmeans.py <datapath>
```

Then it will pop up plot windows and results

# GMM and spectral clustering README

### Procedure

```
1.Go the main class
2. modify parameter
3. run code from ide or command line
```

### Parameter for GMM

```
filename - input file Ln:170
K - number of cluster Ln:172
mu - initial cluster center ln 173
cov - initial covariance matrix ln:174
prior - initial prior prob by user ln:175

more default param in function GMM init (smooth,threshold,max_iter)
```

### Parameter for Spectral

```
filename - input file Ln:98
K - number of cluster Ln:111 or activate 110 for auto finding the optimal k
sigma - initial sigma ln 99


```

### Parameter for k-mean

```
filename - input from command line first arg
all variable can by modify on init function of Kmean class ln 13 or ln 143

```
