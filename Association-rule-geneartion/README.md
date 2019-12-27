## What has been done?
* Apriori algorithm has been used for association rules generation. It includes two major steps
	* Frequent itemset generation based on the frequency threshold. After each level of generation, prune the non-frequent itemsets to avoid further computations related to those itemsets.
	* Rule generation. For each frequent itemset, iterate all its subsets as causal of the rule, the remaining as effects of the rule. Based on confidence meassure, find all rules. 

## How to run?
```bash
python3 Rulegeneration.py <filepath>
```

filepath is the full path of data, whose columns are gene's attributes, rows are samples. Here is Code/associationruletestdata.txt

## Sample of generated rules

{G54_Up}->{G24_Down}
{G54_Up}->{G88_Down}
{G59_Up,G72_Up}->{G82_Down}
{G59_Up,G72_Up}->{G96_Down}
{G59_Up,G82_Down}->{G72_Up}
{G59_Up,G96_Down}->{G72_Up}
