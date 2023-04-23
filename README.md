Download Link: https://assignmentchef.com/product/solved-40-319-homework3-em-algorithm-for-clustering
<br>
In this problem, we will implement the EM algorithm for clustering. Start by importing the required packages and preparing the dataset.

<strong>import </strong>numpy as np <strong>import </strong>matplotlib . pyplot as plt

<strong>from </strong>numpy <strong>import </strong>linalg as LA <strong>from </strong>matplotlib . patches <strong>import </strong>Ellipse <strong>from </strong>sklearn . datasets . samples generator <strong>import </strong>make blobs <strong>from </strong>scipy . stats <strong>import </strong>multivariate normal

K = 3

NUMDATAPTS = 150

X, y = make blobs ( n samples=NUMDATAPTS, centers=K, shuffle=False , random state=0, cluster std =0.6)

g1 = np. asarray ([[2.0 , 0] , [ −0.9 , 1]]) g2 = np. asarray ([[1.4 , 0] , [0.5 , 0 . 7 ] ] ) mean1 = np.mean(X[ : <strong>int </strong>(NUMDATAPTS/K)])

mean2 = np.mean(X[ <strong>int </strong>(NUMDATAPTS/K):2∗ <strong>int </strong>(NUMDATAPTS/K)])

X[ : <strong>int </strong>(NUMDATAPTS/K)] = np. einsum( ’nj , ij −<em>&gt;</em>ni ’ ,

X[ : <strong>int </strong>(NUMDATAPTS/K)] − mean1 , g1) + mean1

X[ <strong>int </strong>(NUMDATAPTS/K):2∗ <strong>int </strong>(NUMDATAPTS/K)] = np. einsum( ’nj , ij −<em>&gt;</em>ni ’ ,

X[ <strong>int </strong>(NUMDATAPTS/K):2∗ <strong>int </strong>(NUMDATAPTS/K)] − mean2 , g2) + mean2

X[ : , 1 ] −= 4

<ul>

 <li>Randomly initialize a numpy array <em>mu </em>of shape (K, 2) to represent the mean of the clusters, and initialize an array <em>cov </em>of shape (K, 2, 2) such that <em>cov</em>[<em>k</em>] is the identity matrix for each <em>k</em>. <em>cov </em>will be used to represent the covariance matrices of the clusters. Finally, set <em>π </em>to be the uniform distribution at the start of the program.</li>

 <li>Write a function to perform the E-step:</li>

</ul>

<strong>def </strong>E step ():

gamma = np. zeros ((NUMDATAPTS, K))

. . .

. . .

<strong>return </strong>gamma

<ul>

 <li>Write a function to perform the M-step:</li>

</ul>

<strong>def </strong>M step(gamma):

. . . . . .

<ul>

 <li>Now write a loop that iterates through the E and M steps, and terminates after the change inlog-likelihood is below some threshold. At each iteration, print out the log-likelihood, and use the following function to plot the progress of the algorithm:</li>

</ul>

<strong>def </strong>plot result (gamma=None ):

ax = plt . subplot (111 , aspect=’ equal ’ ) ax . setxlim ([−5 , 5]) ax . set ylim ([−5 , 5]) ax . scatter (X[: , 0] , X[: , 1] , c=gamma, s=50, cmap=None)

<strong>for </strong>k <strong>in range</strong>(K):

l , v = LA. eig (cov [ k ]) theta = np. arctan (v [1 ,0]/ v [0 ,0])

e = Ellipse ((mu[k , 0] , mu[k , 1]) , 6∗ l [0] , 6∗ l [1] , theta ∗ 180 / np. pi )

e . set alpha (0.5) ax . add artist (e) plt . show()

<ul>

 <li>Use sklearn’s KMeans module</li>

</ul>

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html to perform K-means clustering on the dataset, and compare both clustering results.

<h1>Problem 2</h1>

Let <em>p </em>and <em>q </em>be distributions on {1<em>,</em>2<em>,</em>3<em>,</em>4<em>,</em>5} such that, and

.

<ul>

 <li>Compute the cross-entropy <em>H</em>(<em>p,q</em>) in bits. Is <em>H</em>(<em>q,p</em>) = <em>H</em>(<em>p,q</em>)?</li>

 <li>Compute the entropies <em>H</em>(<em>p</em>) and <em>H</em>(<em>q</em>) in bits.</li>

 <li>Compute the KL-divergence <em>D<sub>KL</sub></em>(<em>p</em>|<em>q</em>) in bits.</li>

</ul>

Show all working and leave your answers in fractions.

<h1>Problem 3</h1>

<ul>

 <li>Perform singular value decomposition (SVD) on the following matrix</li>

</ul>

<em>.</em>

<ul>

 <li>For a general design matrix <em>X</em>, why are the columns of the transformed matrix <em>T </em>= <em>XV </em>orthogonal?</li>

</ul>

<h1>Problem 4</h1>

In this problem, we will perform principal component analysis (PCA) on sklearn’s diabetes dataset. Start by importing the required packages and load the dataset.

<strong>import </strong>numpy as np <strong>from </strong>sklearn <strong>import </strong>decomposition <strong>from </strong>sklearn <strong>import </strong>datasets

X = datasets . load diabetes (). data

You can find out more on how to use sklearn’s PCA module from:

https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

For this problem, make sure the design matrix is first normalized to have zero mean and unit standard deviation for each column.

<ul>

 <li>Write code to print the matrix <em>V </em>that will be used to transform the dataset, and print all the singular values.</li>

 <li>Now perform PCA on the dataset and print out the 3 most important components for the first10 data-points.</li>

</ul>

<h1>Problem 5</h1>

An AR(2) model assumes the form

<em>r</em><em>t </em>= <em>φ</em>0 + <em>φ</em>1<em>r</em><em>t</em>−1 + <em>φ</em>2<em>r</em><em>t</em>−2 + <em>a</em><em>t</em><em>,</em>

where <em>a<sub>t </sub></em>is a white noise sequence. Show that if the model is stationary, then

(assume <em>φ</em><sub>1 </sub>+ <em>φ</em><sub>2 </sub>6= 1);

(b) the ACF is given by

<em>.</em>