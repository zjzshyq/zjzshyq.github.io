### ss
## s
# s

```python
print("number of rules: ", len(associated_rules))
reflect_rec = list((map(lambda x: (x.lhs[0],x.rhs[0]),associated_rules)))
rules_dict = {}
for tup in reflect_rec:
    if tup[0] not in rules_dict:
        rules_dict[tup[0]] = tup
```
dasdsdas