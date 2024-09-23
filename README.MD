### Absorption graph generator for binary and ternary mixtures

This code is a simple absorption graph generator for binary and ternary mixtures. It is written in Python and uses the matplotlib library to generate the graphs.


### Instruction to run the script

1. Clone the repository
2. install python3.11 and toml
3. run ``poetry install`` to install the dependencies
4. for generate graph for binary mixtures run below command
```python plot_binary.py --file 'file path to the csv' --A 'component name' --B 'component name'```
Example:
```python plot_binary.py --file data/2-Components.csv --A CH3Cl --B CH4```
5. for generate graph for ternary mixtures run below command
```python plot_ternary.py --file 'file path to the csv' --A 'component name' --B 'component name' --C 'component name'```
```python plot_ternary.py --file data/3-Components.csv --A CH3Cl --B CH4 --C CCl4```


### Note: csv should be according to the format available in data directory sample ###
