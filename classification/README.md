# Classification methods

## KNN 

### Requirement

```
$ pip install pandas numpy
```

### Run

Test Pass on Python 2.7

```
$ python KNN.py
```

Then it will pop up plot windows and results


## Decision Tree 

```
$ python Decision_tree.py
```

Before executing the program change the parameter

```
filename - input file Ln:325
without cross validation:
uncomment Ln335 - Ln340
with cross validation
uncomment Ln344 - Ln349
```


## Random Forests 

```
$ python Decision_tree.py
```
Before executing the program change the parameter
```
filename - input file Ln:325
without cross validation:
uncomment Ln354 - Ln360
n_feature: number of feature selected in each iteration Ln:350
n_tree: number of tree in forest Ln:351
with cross validation
uncomment Ln364 - Ln370
n_feature: number of feature selected in each iteration Ln:365
n_tree: number of tree in forest Ln:366
```

## Naïve Bayes

```
$ python3 bayes.py <PATH of Dataset>
```
It gives the mean F1 and accuracy for the cross validation of multinomial naive bayes method
