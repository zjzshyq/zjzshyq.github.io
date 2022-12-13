# <center>A quick recall strategy based on clustering centers for a startup in a recommendation system</center>

<center>
<strong>Yiqing Hu</strong>
</br>Faculty of Economic Sciences, University of Warsaw, Warsaw, Poland
</center>

**Abstract:** Startup recommendation for new users without 
any behavior data is always a challenge for the recommendation system. 
We can use several advanced algorithms or strategies to deal with 
the startup in real production. However, in this course, we do not 
have so many tools in the box we discussed in class. Using the k-means 
to cluster old users to get centers, and getting the center with a 
minimum distance from the new user is the feasible solution to 
fulfill our final course task. The clustering center from k-means
delivers the new user recommend list which produced by Apriori. 
PCA reduces the dimensions of user features and assists to
determine the number of k in k-means. 
</br>**Key Words:** K-means, PCA, Apriori, Recommendation System

## 0. Paper Structure
In the chapter Algorithms, we will discuss these three algorithms and 
implement them by code. We will explore the data we use in this paper 
in the chapter on Data. Chapter Clustering shows the analytic process 
and results of getting the centers, which will conclude PCA and 
k-means. Chapter Recommendation discusses how we get the recommend
list using Apriori, and how we combine cluster centers and recommend
lists together to recommend for the new. The Presentation chapter 
is a simple data analysis website I built to explore data visually. 
And we have a Conclusion in the last chapter.

## 1. Algorithms
In this chapter, we will quickly review k-means, PCA, and Apriori algorithms.
### 1.1 K-mean
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
/usr/bin/python3 /Users/huyiqing/PycharmProjects/UW_lab/us_lab1/kmeans_demo.py 
[ 24.5 224.5] [ 74.5 274.5]
[[ 34.64823228 105.3589104    0.        ]
 [ 33.23401872 103.94469683   0.        ]
 [ 31.81980515 102.53048327   0.        ]
 .
 .
 .
 [103.94469683  33.23401872   1.        ]
 [105.3589104   34.64823228   1.        ]]

Process finished with exit code 0
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
/usr/bin/python3 /Users/huyiqing/PycharmProjects/UW_lab/us_lab1/pca_demo.py 
Component importance of PCA in scikit-learn
[4.25200900e-01 1.77231437e-01 1.24532921e-01 7.31860858e-02
 6.93467514e-02 5.38007297e-02 4.12972825e-02 2.58732153e-02
 9.52265378e-03 8.02349649e-06]

Component importance of My PCA
[4.25200900e-01 1.77231437e-01 1.24532921e-01 7.31860858e-02
 6.93467514e-02 5.38007297e-02 4.12972825e-02 2.58732153e-02
 9.52265378e-03 8.02349649e-06]

Process finished with exit code 0
```
## 1.3 Apriori

Apriori<a href="#ref_1">[1]</a> is one of the algorithms of association analysis which is the 
task of finding relationships in large-scale datasets. These relationships 
can take two forms: (1) frequent item sets, and (2) association rules.
### 1.3.1 Frequent item sets
**Frequent item set:** It is a collection of items that often appear together.
</br>**Quantification method - Support:** Support is the proportion of records in the 
dataset that contains the item set. For example, in the data set 
[[1, 3, 4], [2, 3, 5], [1, 2, 3], [2, 5]], the support of the item set 
{2} is 3/4, and the item the support of the set {2,3} is 1/2.

### 1.3.2 Association Rules
**Association rules:** Implying that there may be a strong relationship between two items.
</br>**Quantitative calculation - Confidence:** Confidence is defined for an association rule such as {2}-->{3}. {2}-->{3}, 
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

![不会显示中括号中的文字](./img/apriori_1.jpg)

## Data 
Use user data and item data are downloaded from the Internet. 
They are both CSVs. In this chapter, we will check the data quickly to 
verify if they are qualified to run the jobs for unsupervised learning.
### 2.1 User Data
User data uses travel reviews data set<a href='#ref_2'>[2]</a>  which is a data set for reviewing destinations in 10 categories mentioned across East Asia. Each traveler rating is mapped as Excellent(4), Very Good(3), Average(2), Poor(1), and Terrible(0) and average rating is used. It is populated by crawling TripAdvisor.com. Reviews on destinations in 10 categories mentioned across East Asia are considered. Each traveler rating is mapped as Excellent (4), Very Good (3), Average (2), Poor (1), and Terrible (0) and the average rating is used against each category per user.

### 2.1.1 Attribute Information
Attribute 1: Unique user ID
</br>Attribute 2: Average user feedback on art galleries 
</br>Attribute 3: Average user feedback on dance clubs 
</br>Attribute 4: Average user feedback on juice bars 
</br>Attribute 5: Average user feedback on restaurants 
</br>Attribute 6: Average user feedback on museums 
</br>Attribute 7: Average user feedback on resorts 
</br>Attribute 8: Average user feedback on parks/picnic spots 
</br>Attribute 9: Average user feedback on beaches 
</br>Attribute 10: Average user feedback on theaters 
</br>Attribute 11: Average user feedback on religious institutions

### 2.1.2 Data statistic
Using python pandas to read the CSV file of the data and check the data. Printing 
the result of function Info() shows there is no Missing value and all data are 
float, not category or character which means feature engineering free.

## 2.2 Item Data
Item data uses the groceries dataset<a href="#ref_3">[3]</a> which has 38765 rows of the purchase orders
of people from grocery stores.

### 2.2.1 Attribute Information
Member_number: User ID 
</br>itemDescription: Item name
</br>Date: The data user buys the item

### 2.2.2 Data statistic
Using python pandas to read the CSV file of the data checking the basic 
information such as the number of instances, number of users, number of 
items, and if there are missing values.


## REFERENCES 
<a id='ref_1'>[1]</a> Rakesh Agrawal and Ramakrishnan Srikant Fast algorithms for mining association rules. Proceedings of the 20th International Conference on Very Large Data Bases, VLDB, pages 487-499, Santiago, Chile, September 1994.

<a id='ref_2'>[2]</a> Shini Renjith, UCI Machine Learning Repository, 19 December 2018. Travel Reviews Data Set. https://archive.ics.uci.edu/ml/datasets/Travel+Reviews

<a id='ref_3'>[3]</a> Heeral Dedhia, Kaggle, 2020. Groceries dataset. https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset.

[4] Shepard, Roger N. (1962). "The analysis of proximities: Multidimensional scaling with an unknown distance function. I.". Psychometrika. 27 (2): 125–140. doi:10.1007/BF02289630. S2CID 186222646.

