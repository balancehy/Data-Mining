## What has been done?
* Apriori algorithm has been used for association rules generation. It includes two major steps
	* Frequent itemset generation based on the frequency threshold. After each level of generation, prune the non-frequent itemsets to avoid further computations related to those itemsets.
	* Rule generation. For each frequent itemset, iterate all its subsets as causal of the rule, the remaining as effects of the rule. Based on confidence meassure, find all rules. 

## How to run?
```bash
python3 Rulegeneration.py <filepath>
```

<filepath> is the full path of data, whose columns are gene's attributes, rows are samples.
