# <center>A fast recall strategy based on clustering centers for a startup in a recommendation system</center>

<center>
<strong>Yiqing Hu</strong>
<br>Faculty of Economic Sciences, University of Warsaw, Warsaw, Poland
</center>
<br>

**Abstract:** Startup recommendation for new users without 
any behavior data is always a challenge for the recommendation system. 
We can use several advanced algorithms or strategies to deal with 
the startup in real production which can be a classic task in cell phone
application. However, in this course, we do not 
have so many tools in the box we discussed in class. Using the k-means 
to cluster old users to get centers, and getting the center with a 
minimum distance from the new user is the feasible solution to 
fulfill our final course task. The clustering center from k-means
delivers the new user recommend list which produced by Apriori. 
PCA reduces the dimensions of user features and assists to
determine the number of k in k-means. 
<br>**Keywords:** K-means, PCA, Apriori, Recommendation System


## 0. Structure
In the chapter Algorithms, we will discuss these three algorithms and 
implement them by code. We will explore the data we use in this paper 
in the chapter on Data. Chapter Clustering shows the analytic process 
and results of getting the centers, which will conclude PCA and 
k-means. Chapter Recommendation discusses how we get the recommendation
list[matrix_t_0_troll.csv](..%2F..%2FDownloads%2F%E8%80%83%E8%AF%95%2Fexam_micro%2Ftask%203%2FDeGroot%2Fmatrix_t_0_troll.csv) using Apriori, and how we combine cluster centers and recommend
lists together to recommend for the new. The Results chapter 
is a simple data analysis website I built to explore data visually. 
And we have a Conclusion in the last chapter.

![pic_0](./img/UL_structrue.png)

For the unsupervised learning course, I will use PCA and K-means in chapter 
3.Clustering and Apriori in chapter 4.Recommendation. 
We will deal with the main work and analysis in these two chapters.
Because it is not convenient to separate one for R coding. So I add some R code in chapter 2.Data
mainly exploring item data with Apriori.

## 1. Algorithms
In this chapter, we will quickly review k-means, PCA, and Apriori algorithms.
### 1.1 K-means
The k-means clustering implementation contains 4 steps:
1. Selecting the number of clusters. In my previous work 
dealing with user data that has more than 400,000 DAU. 
It is hard to determine that number, so we usually start this
algorithm with the K of Sqrt(n/2) or use the information 
within the user data like address, hobby, and so on. 
Gap statistics and the Silhouette Coefficient are sometimes used, 
but these cost a lot of time. In a Big Data environment, determining 
the number of K is very important, because it really costs our machine 
to calculate the distance from a node to every k center.
2. Calculating the distance between the vectors of two nodes.  
3. Calculating the new centers by mean.
4. Repeat steps 2 and 3 until getting the maximum number of iterates or
reaching the min error.

Following the steps of k-means, we can easily implement the algorithm with code. There is a snapshot of the python code 
with comments on k-means I wrote below:
```python
import numpy as np
n = 100
x = np.arange(100)
y = np.arange(200, 300, 1)

# 1、set the number of clusters,  chose k (0,0) and (1,1) as the initial center just for demo
k = 2
center0 = np.array([x[0], y[0]])
center1 = np.array([x[1], y[1]])
dist = np.zeros([n, k + 1])  # initialize the distance maxtrix which is just for demo

_iter = 0
_err = None
while _iter<20:
    # 2、using Euclidean to calculate the distance of vectors between two point
    for i in range(n):
        dist[i, 0] = np.sqrt((x[i] - center0[0]) ** 2 + (y[i] - center0[1]) ** 2)
        dist[i, 1] = np.sqrt((x[i] - center1[0]) ** 2 + (y[i] - center1[1]) ** 2)

        # 3、using the calculated distance to decide which cluster the point belong to
        if dist[i, 0] <= dist[i, 1]:
            dist[i, 2] = 0
        else:
            dist[i, 2] = 1

    # 4、calculating the center of clusters

    # 4.1、initializing index
    index0 = dist[:, 2] == 0  # 所有行的第三列为0
    index1 = dist[:, 2] == 1  # 所有行的第三列为1

    # 4.2 using mean to re-index
    center0_new = np.array([x[index0].mean(), y[index0].mean()])
    center1_new = np.array([x[index1].mean(), y[index1].mean()])

    # 5、updating the new center of clusters
    if (center0 == center0_new).all() and (center1 == center1_new).all():
        break
    else:
        center0 = center0_new
        center1 = center1_new
    _iter += 1

print(center0, center1)
print(dist)
```
The result is below. The first print
[ 24.5 224.5] [ 74.5 274.5] are the coordinates of the two centers. 
The second print is the matrix of nodes with the center label which 
is on the third column in which 0 represents the center of 
[ 24.5 224.5] and 1 represents [ 74.5 274.5].

```
[ 24.5 224.5] [ 74.5 274.5]
[[ 34.64823228 105.3589104    0.        ]
 [ 33.23401872 103.94469683   0.        ]
 [ 31.81980515 102.53048327   0.        ]
 .
 .
 .
 [103.94469683  33.23401872   1.        ]
 [105.3589104   34.64823228   1.        ]]
```
## 1.2 PCA
The PCA implementation contains mainly 5 steps:
1. Normalizing original data matrix X. 
2. Calculating the covariance matrix D(X) of X.
3. Calculating the eigenvalues and eigenvectors of matrix D(X).
4. orting and Indexing the eigenvalues related to the original vector.
5. Getting the top k important vectors.

Following the steps of PCA, we can use the package of NumPy to implement the algorithm. There is a snapshot 
of the python code with comments on PCA I wrote below:
```python
def my_pca(array):
    # calculate the cov-variance matrix of the array
    cov = np.cov(array, rowvar=False)

    # calculate the eigenmatrix,
    # each column in the matrix is the eigenvector
    eig = np.linalg.eig(cov)

    # get the components and variance_ratio
    components = eig[1]
    vars = eig[0]
    explained_variance_ratio = vars/sum(vars)
    print(explained_variance_ratio)

    # calculate standard variance to get eigenvalue
    sds = np.sqrt(vars)
    eigenvalues = components * sds

    # sorting and decreasing eigenvalue
    sorted_eig_val = eigenvalues[:, np.argsort(-vars)].T
    return sorted_eig_val, explained_variance_ratio

# use PCA in the pkg of scikit-learn
pca = PCA()
pca.fit(mtx)
pca_ratio = pca.explained_variance_ratio_

print('Component importance of PCA in scikit-learn')
print(pca_ratio)
print('')
print('Component importance of My PCA')
my_pca(mtx)[1]
```
Comparing the results of the snapshot below from PCA of my code with NumPy and 
PCA in scikit-learn, we can notice that the variance ratio is the same proving 
our code works.

```
Component importance of PCA in scikit-learn
[4.25200900e-01 1.77231437e-01 1.24532921e-01 7.31860858e-02
 6.93467514e-02 5.38007297e-02 4.12972825e-02 2.58732153e-02
 9.52265378e-03 8.02349649e-06]

Component importance of My PCA
[4.25200900e-01 1.77231437e-01 1.24532921e-01 7.31860858e-02
 6.93467514e-02 5.38007297e-02 4.12972825e-02 2.58732153e-02
 9.52265378e-03 8.02349649e-06]
```
## 1.3 Apriori

Apriori<a href="#ref_1">[1]</a> is one of the algorithms of association analysis which is the 
task of finding relationships in large-scale datasets. These relationships 
can take two forms: (1) frequent item sets, and (2) association rules.
### 1.3.1 Frequent item sets
**Frequent item set:** It is a collection of items that often appear together.
<br>**Quantification method - Support:** Support is the proportion of records in the 
dataset that contains the item set. For example, in the data set 
[[1, 3, 4], [2, 3, 5], [1, 2, 3], [2, 5]], the support of the item set 
{2} is 3/4, and the item the support of the set {2,3} is 1/2.

### 1.3.2 Association Rules
**Association rules:** Implying that there may be a strong relationship between two items.
<br>**Quantitative calculation - Confidence:** Confidence is defined for an association rule such as {2}-->{3}. {2}-->{3}, 
the reliability of this rule is “support_degree{2, 3}/support_degree{2}”, that is, 2/3, which means 2/3 in all records containing 
{2} in {2,3} with the rules.

### 1.3.3 Principle of Apriori
If an item set is frequent, then all its subsets are also frequent.
Conversely, if an item set is infrequent, then all its supersets are 
also infrequent. Based on these principles, Apriori can avoid the exponential 
growth of the number of item sets, so that frequent item sets can be calculated 
in a reasonable time.

### 1.3.4 Apriori algorithm process
1. First, generate a list C1 of item sets with several 1 based on the data set. 
2. According to the frequent item set function, calculate the support degree of each 
element in C1, remove the elements that do not meet the minimum support degree, and generate the frequent item set list L1 that meets the minimum support degree.
3. Generate a candidate item set list C2 with k=2 based on L1 according to the 
function of creating candidate item sets.
4. According to the frequent item set function, based on C2, generate a frequent 
item set list L2 that satisfies the minimum support degree k=2.
5. Increasing the value of k, repeat 3) and 4) to generate Lk until Lk is empty, 
return the L list, L includes L1, L2, L3…

The sample handwriting Apriori process is below:

![pic_1.1](./img/apriori_1.jpg)

## 2. Data 
Use user data and item data are downloaded from the Internet. 
They are both CSVs. In this chapter, we will check the data quickly to 
verify if they are qualified to run the jobs for unsupervised learning.
### 2.1 User Data
User data uses travel reviews data set<a href='#ref_2'>[2]</a>  
which is a data set for reviewing destinations in 10 categories mentioned 
across East Asia. Each traveler rating is mapped as Excellent(4), Very Good(3), 
Average(2), Poor(1), and Terrible(0) and average rating is used. 
It is populated by crawling TripAdvisor.com. 
Reviews on destinations in 10 categories mentioned across
East Asia are considered. Each traveler rating and the average rating 
is used against each category per user.

### 2.1.1 Attribute Information
Attribute 0: Unique user ID
<br>Attribute 1: Average user feedback on art galleries 
<br>Attribute 2: Average user feedback on dance clubs 
<br>Attribute 3: Average user feedback on juice bars 
<br>Attribute 4: Average user feedback on restaurants 
<br>Attribute 5: Average user feedback on museums 
<br>Attribute 6: Average user feedback on resorts 
<br>Attribute 7: Average user feedback on parks/picnic spots 
<br>Attribute 8: Average user feedback on beaches 
<br>Attribute 9: Average user feedback on theaters 
<br>Attribute 10: Average user feedback on religious institutions
```
  User.ID Category.1 Category.2 Category.3 Category.4 Category.5 Category.6 Category.7 Category.8 Category.9 Category.10
1  User 1       0.93       1.80       2.29       0.62       0.80       2.42       3.19       2.79       1.82        2.42
2  User 2       1.02       2.20       2.66       0.64       1.42       3.18       3.21       2.63       1.86        2.32
3  User 3       1.22       0.80       0.54       0.53       0.24       1.54       3.18       2.80       1.31        2.50
4  User 4       0.45       1.80       0.29       0.57       0.46       1.52       3.18       2.96       1.57        2.86
5  User 5       0.51       1.20       1.18       0.57       1.54       2.02       3.18       2.78       1.18        2.54
6  User 6       0.99       1.28       0.72       0.27       0.74       1.26       3.17       2.89       1.66        3.66
```

### 2.1.2 Data statistic
Using R Language to read the CSV file of the data and check the data. Printing 
the results with the R functions shows there is no Missing value and all data 
except User.ID are numeric, not category or character which means do not need
feature engineering.

```R
> sapply(tripadvisor, class)
    User.ID  Category.1  Category.2  Category.3  Category.4  Category.5  Category.6  Category.7 
"character"   "numeric"   "numeric"   "numeric"   "numeric"   "numeric"   "numeric"   "numeric" 
 Category.8  Category.9 Category.10 
  "numeric"   "numeric"   "numeric" 
  
> tripadvisor[!complete.cases(tripadvisor),]
 [1] User.ID     Category.1  Category.2  Category.3  Category.4  Category.5  Category.6  Category.7 
 [9] Category.8  Category.9  Category.10
<0 rows> (or 0-length row.names)
```

Function summary() in R shows all features which are on a similar scale proving there’s no need to 
transform the data. 

```R
 User.ID            Category.1       Category.2      Category.3      Category.4       Category.5       Category.6      Category.7      Category.8      Category.9     Category.10   
 Length:980         Min.   :0.3400   Min.   :0.000   Min.   :0.130   Min.   :0.1500   Min.   :0.0600   Min.   :0.140   Min.   :3.160   Min.   :2.420   Min.   :0.740   Min.   :2.140  
 Class :character   1st Qu.:0.6700   1st Qu.:1.080   1st Qu.:0.270   1st Qu.:0.4100   1st Qu.:0.6400   1st Qu.:1.460   1st Qu.:3.180   1st Qu.:2.740   1st Qu.:1.310   1st Qu.:2.540  
 Mode  :character   Median :0.8300   Median :1.280   Median :0.820   Median :0.5000   Median :0.9000   Median :1.800   Median :3.180   Median :2.820   Median :1.540   Median :2.780  
                    Mean   :0.8932   Mean   :1.353   Mean   :1.013   Mean   :0.5325   Mean   :0.9397   Mean   :1.843   Mean   :3.181   Mean   :2.835   Mean   :1.569   Mean   :2.799  
                    3rd Qu.:1.0200   3rd Qu.:1.560   3rd Qu.:1.573   3rd Qu.:0.5800   3rd Qu.:1.2000   3rd Qu.:2.200   3rd Qu.:3.180   3rd Qu.:2.910   3rd Qu.:1.760   3rd Qu.:3.040  
                    Max.   :3.2200   Max.   :3.640   Max.   :3.620   Max.   :3.4400   Max.   :3.3000   Max.   :3.760   Max.   :3.210   Max.   :3.390   Max.   :3.170   Max.   :3.660   
```
Function parirplot() in the python from package seaborn can print the 
correlation figure  between two features. The diagonal is the distribution of the feature itself. 
Feature 4 and feature 7 are not paired so well with other features. We will deal with them in the 
next chapter.

![pic_2.1](./img/pairplot.png)

### 2.1.3 Data distribution
Using R to detect the distribution of user data.
```R
trips<-tripadvisor[,2:11]
dist1<-dist(trips,method = 'euclidean')
mds_fitted<-mds(dist1, ndim=2,  type="ratio")
stress<-mds_fitted$stress
plot(mds_fitted$conf, main="MDS for all users", pch=20)
```
The average stress is 0.2022141 which is acceptable. The distribution based on MDS with
euclidean distance between users are shown as below. Most of the users gather
together, not separate too much with the distance of two dimensions. 
Furthermore, users at the right of the red line are more centralized than the left ones,
meaning users for some trip spots have consistent preferences that may be commonly liked. 
Users at the right line tend to have wider choices. The outlier exists, of course, small 
tourist attractions will always attract some people.

![pic_2.2](./img/MDS4users.png)

After analysis the MDS result, we select 5 as cluster number for convex clustering, 
we can find users in these clusters represent travelling preferences are not so dispersed.
```R
cc<-cclust(trips, 5, dist="euclidean")
stripes(cc)
```
![pic_2.2](./img/cclust.png)



## 2.2 Item Data
Item data uses the groceries dataset<a href="#ref_3">[3]</a> which has 38765 rows of the purchase orders
of people from grocery stores.

### 2.2.1 Attribute Information
Member_number: User ID 
<br>itemDescription: Item name
<br>Date: The data user buys the item
```
  Member_number       Date  itemDescription
1          1808 21-07-2015   tropical fruit
2          2552 05-01-2015       whole milk
3          2300 19-09-2015        pip fruit
4          1187 12-12-2015 other vegetables
5          3037 01-02-2015       whole milk
6          4941 14-02-2015       rolls/buns
```

### 2.2.2 Data statistic
Using R language to read the CSV file of the data checking the basic 
information such as the number of instances, number of users, number of 
items, and if there are missing values.
```R
num_instances <- nrow(groceries)
num_user <- length(unique(groceries$Member_number))
num_item <- length(unique(groceries$itemDescription))
na_check <- groceries[!complete.cases(groceries), ]
```
```
[1] 38765
[1] 3898
[1] 167
[1] Member_number   Date            itemDescription
<0 rows> (or 0-length row.names)
```

Observing the frequency of grocery items, it is easy to find that whole milk, 
other vegetables are the most two favourite products for consumers.
```R
itemsets <- apriori(Groceries, 
                    parameter=list(minlen=1, 
                                   maxlen=1, 
                                   support=0.05, 
                                   target="frequent itemsets"))
inspect(
  head(
    sort(itemsets, 
         by = c("support", 'count')), 
    10))
```
```
     items              support    count
[1]  {whole milk}       0.25551601 2513 
[2]  {other vegetables} 0.19349263 1903 
[3]  {rolls/buns}       0.18393493 1809 
[4]  {soda}             0.17437722 1715 
[5]  {yogurt}           0.13950178 1372 
[6]  {bottled water}    0.11052364 1087 
[7]  {root vegetables}  0.10899847 1072 
[8]  {tropical fruit}   0.10493137 1032 
[9]  {shopping bags}    0.09852567  969 
[10] {sausage}          0.09395018  924 
```
![pic_2.3](./img/item_distr1.png)
![pic_2.4](./img/item_distr2.png)

### 2.2.3 Item Exploration
In this sector, we use Apriori in R to explore the association of 
grocery items by visualization.
```R
rules <- apriori(Groceries, 
                 parameter=list(support=0.05,
                                confidence=0.1, 
                                target = "rules"))
inspect(
  head(
    sort(rules, 
         by=c('lift',"support", 'confidence'), 
         decreasing=T), 
    10))
```
We can get association rules from Apriori in the code above. And the rules data table is 
below which contains columns of lhs(rule pre-rule item) and rhs((rule post-item). The lhs 
indicates the condition that needs to be met to trigger the rule, and the rhs indicates the 
expected result after the condition is met. The column lift here means the degree of 
improvement, and describes the strong and weak association of this rule. Lift>1 means 
strong association, and vice versa.

For example about rhs, lhs and lift, we will buy whole milk after yogurt purchase, 
and there is a strong association on the order of purchase.
```
     lhs                   rhs                support    confidence coverage  lift     count
[1]  {yogurt}           => {whole milk}       0.05602440 0.4016035  0.1395018 1.571735  551 
[2]  {whole milk}       => {yogurt}           0.05602440 0.2192598  0.2555160 1.571735  551 
[3]  {other vegetables} => {whole milk}       0.07483477 0.3867578  0.1934926 1.513634  736 
[4]  {whole milk}       => {other vegetables} 0.07483477 0.2928770  0.2555160 1.513634  736 
[5]  {rolls/buns}       => {whole milk}       0.05663447 0.3079049  0.1839349 1.205032  557 
[6]  {whole milk}       => {rolls/buns}       0.05663447 0.2216474  0.2555160 1.205032  557 
[7]  {}                 => {rolls/buns}       0.18393493 0.1839349  1.0000000 1.000000 1809 
[8]  {}                 => {yogurt}           0.13950178 0.1395018  1.0000000 1.000000 1372 
[9]  {}                 => {whole milk}       0.25551601 0.2555160  1.0000000 1.000000 2513 
[10] {}                 => {other vegetables} 0.19349263 0.1934926  1.0000000 1.000000 1903 
```

Then we get the rule plot on confidence and support with lift. 
Item pairs with strong association(lift>1), Their confidence and support 
gather in the upper right corner of this plot. There are not so many strong 
associations with high support in grocery purchase behaviour.

In the second plot, we can get the correlation between the two parameters. 
Like support and confidence, they have positive a relation to the rules.
```
plot(rules)
plot(rules@quality)
```
![pic_2.5](./img/scatter_lift_conf_sup.png)
![pic_2.6](./img/scatters_corr_conf_sup.png)

The plot of the matrix for rules with confidence>1 and strong associations 
shows the distribution of the items with rhs and lhs. For example, yogurt in 
rules data table has 1 lhs and 8 rhs, and it appears on the top left of the graph.
```R
confidentRules <- rules[quality(rules)$confidence > 0.1]
plot(confidentRules, method="matrix", 
     measure=c("lift", "confidence"), 
     control=list(recorder=TRUE))
```
![pic_2.7](./img/mtx_lift_conf.png)

We can get top 5 lift in the rules by the codes below.
```R
highLiftRules <- head(sort(rules, by="lift"), 5)
plot(highLiftRules, method="graph", control=list(type="items"))
```
And the plot shows a weak association that we would buy whole milk after buying 
rolls/buns. And we always buy whole milk and other vegetables together. 
Yogurt and milk are another paired choice.
![pic_2.8](./img/top5_lift.png)

## 3. Clustering
In this chapter, we will use PCA to reduce the dimension of 
user data which are the average user feedback score. With the reduced data, we run k-means multi-iterations to calculate 
the silhouette score in each iteration shown by a figure to determine the number 
cluster which is K. Plotting the node with the top 3 components in the 3D figure 
to check if the de-dimensioned K-means works well.

### 3.1 Feature reduction
Using the package PCA() for user data check the components. From the result, 
we can find top 3 components are 0.42, 0.17, and 0.12 respectively which make up 
the vast majority ratio. These components make the most contributions to the 
average feedback score. And Components 8 and 9 could be trimmed, which means we
can remove 2 feedback score features that are not so useful in this task.

```python
pca = PCA()
pca.fit(trip_df.values)
n_pcs= pca.components_.shape[0]

pca_ratio = pca.explained_variance_ratio_
print('Component Ratio')
for i, comp in enumerate(pca_ratio):
    print("component_"+str(i+1), "%.5f"%comp)
```
```
Component Ratio
component_1 0.42520
component_2 0.17723
component_3 0.12453
component_4 0.07319
component_5 0.06935
component_6 0.05380
component_7 0.04130
component_8 0.02587
component_9 0.00952
component_10 0.00001
```
It should be emphasized that PCA is not good at feature dropping and selecting. 
Though each component which is an eigenvector contains weights that represent 
the importance of each feature, it is tricky to tell how important the whole 
matrix is. Usually, we can simply use linear regression in a supervised learning 
task to get the features’ weights which determine the output value Y. In this 
paper, we try to use PCA to pick features by the values with the largest 
proportion in each eigenvector.

```python
important_ratio_bottom = []
important_ratio_height = []
feature_acc = [0]*n_pcs
print('\nMost Important Feature in each Component')
for j, i in enumerate(range(n_pcs)):
    component_arr = np.abs(pca.components_[i])
    idx = component_arr.argmax()
    component_ratio = component_arr[idx]/sum(component_arr)
    print('component_'+str(j+1), 'feature_'+str(idx+1), "%.5f"%component_ratio)
    important_ratio_bottom.append(pca_ratio[i]*component_ratio)
    important_ratio_height.append(pca_ratio[i]*(1-component_ratio))
    for k,v in enumerate(component_arr):
        feature_acc[k] += v
```
```
Most Important Feature in each Component
component_1 feature_3 0.44760
component_2 feature_6 0.26113
component_3 feature_2 0.46280
component_4 feature_9 0.49582
component_5 feature_1 0.24801
component_6 feature_10 0.27162
component_7 feature_5 0.22963
component_8 feature_10 0.33426
component_9 feature_8 0.77743
component_10 feature_7 0.97385
```
The result of the first line means that feature 3 with the highest proportion in 
component 1 is 0.44. Feature 3 represents the average score of  juice bars  which means people 
from East Asia are keen on the travelling destinations with juice bars that 
may enhance the travel experience.

The figure below combines two results above and directly shows the ratio of
each component and the ratio of the feature value in each component. 

![pic_3.1](./img/pca_1.jpg)

Then we plot the figure to show the accumulation of all feature values in the 
whole matrix. Feature 7 and 8 are the least, and they are the most important 
feature for components 9 and 10 having the least ratios.

Because we are going to remove Components 9 and 10. Features 8 and 7 can be removed
from our task for making the most contribution to Components 9 and 10. Feature 7 
is average user feedback on parks/picnic spots, feature 8 is on beaches.

From above analysis, beaches and spots of parks/picnics would not be considered so much for 
East Asia travellers. Comparing the correlation figure in paragraph 2.1.2, 
spots of parks/picnics have no correlation with other destinations. 
Beaches may be just not so attractive to East Asia tourist compared with other destinations.

![pic_3.2](./img/pca_2.jpg)

Though the accumulation feature 4 is high, feature 4 is not the top-valued feature 
in each eigenvector. Feature 4 represents the average score of restaurants. 
In our common sense, we usually don’t regard restaurants as a destination for travelling, 
however, restaurants may appear in most tourist destinations. 
It makes sense to remove feature 4 from our data. Feature 4 in paragraph 2.1.2, is not 
paired so well with other features explaining that restaurants are more independent.

So we decide to drop feature 4, 7, and 8 which are restaurants, beaches and parks/picnic spots.


## 3.2 K-selecting
We drop feature  4, 7, and 8 to get new data set for training k-means. The 4 
comments in the function of my_kmeans() show how to code and run k-means 
algorithms in the scikit-learn package. We run my_kmeans() with a k number 
from 2 to 20 to calculate the silhouette score for selecting the number of clusters.

```python
pca_df = trip_df.drop(['Category_7','Category_8', 'Category_4'], axis=1)
def my_kmeans(data_df, k):
    # 1. initialize the model
    my_kmeans = KMeans(n_clusters=k, random_state=1)

    # 2. fit the model to the data
    my_kmeans.fit(data_df)

    # 3. obtain cluster labels
    clusters = my_kmeans.predict(data_df)

    # 4. get cluster centers
    centroids = my_kmeans.cluster_centers_
    return clusters, pd.DataFrame(centroids)

sil_score_lst = []
k_lst = list(range(2,20))
for k in k_lst:
    clusters, centroids_df = my_kmeans(pca_df,k)
    if k == 4:
        trip_df['cluster_label'] = clusters
    score = silhouette_score(pca_df, clusters, metric='euclidean')
    sil_score_lst.append(score)
```
When K=2, the silhouette score gets the highest, and the plot of MDS in chapter2 
proved visually that 2 cluster may be a good choose. So normally, selecting 2 clusters 
is a best choice. But our task is the recommendation, too small of the K is 
not good for recommending which will reduce the richness and difference of 
recommendation. When K=4, the score gets sable, so we choose 4 clusters for our 
remaining tasks. Actually, choosing 4-7 are all accepted, only depending on the 
task scene.

![pic_3.3](./img/sil_1.jpg)

After clustering with K=4, we plot the nodes with the top 3 components in the 
3D nodes scatter figure. 

These 4 clusters are people from East Asia representing travelling destination 
preferences with juice bars, dance clubs and resorts. Travellers in the blue cluster 
do not like these 3 destinations but can accept resorts at least. Travellers in the 
green cluster are keen on resorts and someone thinks dance clubs are also not bad choices. 
Travellers in the red cluster like both resorts and juice bars. Travellers in the yellow 
cluster may choose resorts as their first priority.

![pic_3.4](./img/3d_1.jpg)

Dropping feature 3, 6, and 2 which are the most important features for the top 
3 components will get the silhouette score figure and nodes scatter figure below 
showing the instability of the silhouette score and the implicit result of nodes scatter.

![pic_3.5](./img/sil_3d_2.jpg)

Dropping feature 9,1,10 which are not so important and not so unimportant 
features will get the silhouette score and nodes scatter figure below. The 
results get better than the second one 
but are not as good as the first situation.

![pic_3.6](./img/sil_3d_3.jpg)

Till now, we select 4 clusters with dropped feature 7,8,4 to do the rest job.

The reason why we use PCA to reduce the dimensions is to optimize the cost of 
implement time. Because in real Internet task, it is normal that the number of 
features would be thousands. Dedimension will also improve the situation of 
overfitting for supervised learning. It is quite only for us to check the data by
poltting them on the figure which rely on 2 or 3 dimensions with the PCA components.

## 4. Recommendation
This chapter contains the user recommendation and the item recommendation. 
We will use cluster center getting from k-means to recommend the people to the 
new user who may like which based on the similarity of mental distance 
approaches<a href='#ref_4'>[4]</a>. Item recommendation delivered by Apriori 
recommends new user items they may like. The flow chart below is the whole process
of the strategy to recommend for new user.

![pic_4.1](./img/clustering_rec.jpg)

The red dotted line box does the work as we discussed in Chapter Clustering.
The green line is for new user to get the data for recommend. 
The blue dotted line box is the recommendation list of the genarate module.

### 4.1 User recommendation
If the recommendation task is for new users, the user profile we use in the red dotted line box
will only choose the features that new users would have, like the brand of mobile phone, ip
address, gender, application installed list(som countries can't get this, because of the law), 
and so on. We can't use the user behavior as the features for clustering as we did in the chapter
clustering, because new user do not have this data.

As the flow chart shows, the interface gets new user basic data, then we assemble the data into
a vector which have the same form as cluster centers we already made. We calculate the similarity
with the vectors of cluster centers to get the closest one, then calculate the similarity with users
in the cluster. We can get a list of similar users to recommend the new one.

### 4.2 Item recommendation
We will use groceries dataset csv as original data to implement this task to recommend.
After grouping the items by users from groceries data, each user gets the list of items he clicked.

```python
groceries_grouped = groceries.groupby('Member_number')['itemDescription']\
    .agg(lambda x: ','.join(x))\
    .reset_index(name ='Item_series')
input_items = list(map(lambda x: tuple(set(x.split(','))),
                       list(groceries_grouped['Item_series'])
                       ))
```
The output below is the result from the code above. It is the examples which we 
will regard as  the input data for apriori.
```
input_items sample: 
[('sausage', 'hygiene_articles', 'salty_snack', 'misc._beverages', 'whole_milk', 'pastry', 'canned_beer', 'soda', 'yogurt', 'semi-finished_bread', 'pickled_vegetables'), 
 ('sausage', 'whole_milk', 'soda', 'beef', 'whipped_sour_cream', 'rolls_buns', 'frankfurter', 'white_bread', 'curd'), 
 ('frozen_vegetables', 'whole_milk', 'other_vegetables', 'sugar', 'tropical_fruit', 'butter_milk', 'butter', 'specialty_chocolate')
 ]
```
<br>After we putting the data above into the apriori, and setting the parameters of min_support and min_confidence 
can get the item frequency set and the association rules dictionary.
```python
item_dict, rules4all = apriori(input_items, min_support=0.1, min_confidence=0.11)
reflect_rec = list((map(lambda x: (x.lhs[0],x.rhs[0]),rules4all)))
rules_dict = {}
for tup in reflect_rec:
    if tup[0] not in rules_dict:
        rules_dict[tup[0]] = [tup[1]]
    else:
        rules_dict[tup[0]].append(tup[1])
print('rules_dictionary', rules_dict)
```

Using Apriori to get the associate rules below indicates people tend to buy bottled water 
after buying whole milk. So we store the dictionary of rules in a memory database like Redis.
When users click a certain item, our system will access Redis to get the dictionary for 
recommending items below.
```
rules_dictionary
 {'whole_milk': ['bottled_water', 'other_vegetables', 'rolls_buns', 'root_vegetables', 'sausage', 'soda', 'tropical_fruit', 'yogurt'], 
 'bottled_water': ['whole_milk'], 
 'rolls_buns': ['other_vegetables', 'soda', 'whole_milk', 'yogurt'], 
 'other_vegetables': ['rolls_buns', 'soda', 'whole_milk', 'yogurt'], 
 'soda': ['other_vegetables', 'rolls_buns', 'whole_milk'], 
 'yogurt': ['other_vegetables', 'rolls_buns', 'whole_milk'], 
 'root_vegetables': ['whole_milk'], 'sausage': ['whole_milk'], 
 'tropical_fruit': ['whole_milk']
 }
```
<br>In each cluster, we imply the apriori for the recommend list which contains top 10 frequency items. 
As the flow chart at the beginning of this sector, users from interface will check which cluster
he is most similar with, and then get recommend list of this cluster. When he clicks some item having
the association rules, the system can continually push the item recommend.
```python
for k in set(trip_df['cluster_label']):
    tmp_df = trip_df[trip_df['cluster_label'] == k]
    input_items = list(map(lambda x: tuple(set(x.split(','))),
                           list(tmp_df['Item_series'])
                           ))
    # run Apriori
    item_dict, association_rules= apriori(input_items, min_support=0.1, min_confidence=0.15)
    # clean the frequent item sets
    kv_pairs = map(lambda x: (x[0][0], x[1]), item_dict[1].items())
    # order the item sets
    sorted_paired = sorted(kv_pairs, key=lambda kv: -kv[1])
    # get the most 10 frequent items as the recommend list
    rec_lst_top10 = list(map(lambda x: x[0], sorted_paired))[:10]
    print("cluster %d rec lst top10: " % k, rec_lst_top10)
```

These 4 recommendation lists show that the groups clustered by people with different travelling 
spot preferences have different shopping preferences. However, some products like whole_milk with 
high bought frequency are a favourite choice for most people no matter which clusters they 
are in. In clusters 1 and 2, most recommend products are the same, which suggests that 
these two clusters can be combined into one in this recommendation scenario.
```
cluster 0 rec lst top10:  ['whole_milk', 'other_vegetables', 'rolls_buns', 'yogurt', 'soda', 'root_vegetables', 'bottled_water', 'tropical_fruit', 'pastry', 'bottled_beer']
cluster 1 rec lst top10:  ['whole_milk', 'other_vegetables', 'rolls_buns', 'yogurt', 'soda', 'root_vegetables', 'bottled_water', 'tropical_fruit', 'pip_fruit', 'citrus_fruit']
cluster 2 rec lst top10:  ['whole_milk', 'other_vegetables', 'rolls_buns', 'soda', 'yogurt', 'tropical_fruit', 'root_vegetables', 'bottled_water', 'sausage', 'citrus_fruit']
cluster 3 rec lst top10:  ['whole_milk', 'rolls_buns', 'yogurt', 'other_vegetables', 'tropical_fruit', 'soda', 'root_vegetables', 'sausage', 'bottled_water', 'shopping_bags']
```


## 5. Results
Data Visualization platform is an important part of recommendation system for Presentation.
The platform can very well help data analysts to carry out their work. We bulid a demo of the 
platform to show the result of unsupervised learning. 
This chapter we will introduce every page of the demo.

### 5.1 Login Page
Managers which are users of platform with different permissions have different scopes of use. 
We divide users into normal managers and super managers.

![pic_5.1](./img/page_login.jpg)

### 5.2 Algorithm page
This page lists all the subpages we can access for visualization.

![pic_5.2](./img/page_lists.jpg)

### 5.3 PCA Page 1
Two figures could help us to look into the importance of each feature in the whole data 
and within the component.

![pic_5.3](./img/page_pca.jpg)

### 5.4 PCA Page 2
This page shows which is the most important or has the highest ratio feature in the 
component. We can find the feature names from this list.

![pic_5.4](./img/page_pca_component.jpg)

### 5.5 K-means feature selecting page
In this page, we can choose which feature we want to drop.

![pic_5.5](./img/page_features.jpg)

### 5.6 K-means result page
Figure 1 in this page shows the silhouette score with different number of clusters. 
Figure 2 scatters the nodes in 3D. The coordinates of the 3D is top 3 important features. 
In this page we can select the features to check if it is the result we want.

![pic_5.6](./img/page_kmean_result.jpg)

### 5.7 Apriori parameter page
Setting the parameters of apriori in this page.

![pic_5.7](./img/page_apriori_param.jpg)

### 5.8 Apriori result page
After setting the parameters, we can get the association rules 
from apriori in this page.

![pic_5.8](./img/page_apriori_rules.jpg)


## 6. Conclusion
We combine three paper into one, and add some R content in this paper.
Hoping it is ok to pass the course. This conclusion chapter is just a formality to
complete for the entire paper. Let's end up here. Thanks for reading.

## REFERENCES 
<a id='ref_1'>[1]</a> Rakesh Agrawal and Ramakrishnan Srikant Fast algorithms for mining association rules. Proceedings of the 20th International Conference on Very Large Data Bases, VLDB, pages 487-499, Santiago, Chile, September 1994.

<a id='ref_2'>[2]</a> Shini Renjith, UCI Machine Learning Repository, 19 December 2018. Travel Reviews Data Set. https://archive.ics.uci.edu/ml/datasets/Travel+Reviews

<a id='ref_3'>[3]</a> Heeral Dedhia, Kaggle, 2020. Groceries dataset. https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset.

<a id='ref_4'>[4]</a> Shepard, Roger N. (1962). "The analysis of proximities: Multidimensional scaling with an unknown distance function. I.". Psychometrika. 27 (2): 125–140. doi:10.1007/BF02289630. S2CID 186222646.