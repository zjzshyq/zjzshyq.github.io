# A quick recall strategy based on clustering centers for a startup in a recommendation system
Katarzyna Kopczewska, Yiqing Hu(455858)
Faculty of Economic Sciences, University of Warsaw, Warsaw, Poland

Abstract: Startup recommendation for new users without any behavior data is always a challenge for the recommendation system. We can use several advanced algorithms or strategies to deal with the startup in real production. However, in this course, we do not have so many tools in the box we discussed in class. Using the k-means to cluster old users to get centers, and getting the center with a minimum distance from the new user is the feasible solution to fulfill our final course task. The clustering center from k-means delivers the new user recommend list which produced by Apriori. PCA reduces the dimensions of user features and assists to determine the number of k in k-means. 
Key Words: K-means, PCA, Apriori, Recommendation System

```python
print("number of rules: ", len(associated_rules))
reflect_rec = list((map(lambda x: (x.lhs[0],x.rhs[0]),associated_rules)))
rules_dict = {}
for tup in reflect_rec:
    if tup[0] not in rules_dict:
        rules_dict[tup[0]] = tup
```


