from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

def display_SVs(kernel_type, C_opt, deg = 1, gamma_opt=0):
    if kernel_type == 'poly':
        clf = svm.SVC(C=C_opt,kernel='poly',degree=deg,coef0=1.0)
    elif kernel_type == 'rbf':
        clf = svm.SVC(C=C_opt,kernel='rbf',gamma=gamma_opt)

    clf.fit(X_train,y_train)

    SVs = clf.support_vectors_
    alpha = clf.dual_coef_

    xi = 1 - np.squeeze(np.sign(alpha))*clf.decision_function(SVs)

    origidx = clf.support_

    sortidx = np.argsort(xi)

    topSVs = SVs[sortidx[-16:],:]
    topSVlabels = y_train[origidx[sortidx[-16:]]]

    f, axarr = plt.subplots(4, 4)
    ii=0
    jj=0
    for idx in range(16):
        axarr[ii, jj].imshow(topSVs[idx].reshape((28,28)), cmap='gray')
        axarr[ii, jj].axis('off')
        axarr[ii, jj].set_title('{label}'.format(label=int(topSVlabels[idx])))
        ii = ii + 1
        if ii==4:
            ii = 0
            jj = jj+1

    filename = ''
    if kernel_type == 'poly':
        if deg == 1:
            filename += 'lin'
        elif deg == 2:
            filename += 'quad'
    elif kernel_type == 'rbf':
        filename += 'rbf'

    filename += 'SVs.png'

    plt.savefig(filename)
    plt.show()


np.random.seed(2022)

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data
y = mnist.target.astype(int)   #default return type is a np.array of str

#View the jth image
'''
j = 1
plt.title('The jth image is a {label}'.format(label=int(y[j])))
plt.imshow(X[j].reshape((28,28)), cmap='gray')
plt.show()
'''

X4 = X[y==4,:]
X9 = X[y==9,:]

#######################################################
#######################################################
##              TODO: PART A                         ##
#######################################################
#######################################################




#Create training/validation/testing sets

#######################################################
#######################################################
##              TODO: PART B                         ##
#######################################################
#######################################################
Cvec = ??? #Choose values of C to sweep.

for deg in [1,2]:
    print("Training SVM with polynomial kernel, deg = " + str(deg))
    Pe_train = []   #Probability of error for training set
    Pe_val = []     #Probability of error for validation set
    Pe_test = []    #Probability of error for testing set
    numSvs = []     #Number of support vectors

    for Cval in Cvec:
        print('Testing Cval = ' + str(Cval))
        #Train SVM w/ polynomial kernel using Cval and deg
        #You should compute the probability of error on all 3 sets and store them appropriately.
        #You should also track the number of support vectors.

    #Based on the validation set, set the value for C
    Cval_opt = Cvec[np.argmin(Pe_val)]
    print('Optimal value of C based on validation set: ' + str(Cval_opt))
    print('Train error: ' + str(Pe_train[np.argmin(Pe_val)]))
    print('Test error: ' + str(Pe_test[np.argmin(Pe_val)]))
    print('Number of support vectors: ' + str(numSvs[np.argmin(Pe_val)]))

    #Visualize how training/validation/testing error and # of SVs changes with C
    print('Pe_train: ', Pe_train)
    print('Pe_val: ', Pe_val)
    print('Pe_test: ', Pe_test)
    print('numSvs: ', numSvs)

    #Display the SVs that are hardest to classify
    #THIS IS A BLOCKING OPERATION. For the code to continue, close the figure that is generated
    display_SVs('poly', Cval_opt, deg = deg)


#######################################################
#######################################################
##              TODO: PART C                         ##
#######################################################
#######################################################
Pe_train = []   #Probability of error for training set
Pe_val = []     #Probability of error for validation set
Pe_test = []    #Probability of error for testing set
numSvs = []     #Number of support vectors

Gammavec = ??? #Choose values of gamma to sweep.

for Gammaval in Gammavec:
    print('Testing Gammaval = ' + str(Gammaval))
    #Train SVM w/ rbf kernel using C = 10 and Gammaval
    #You should compute the probability of error on all 3 sets and store them appropriately.
    #You should also track the number of support vectors.

Gammaval_opt = Gammavec[np.argmin(Pe_val)]
print('Optimal value of Gamma based on validation set: ' + str(Gammaval_opt))
print('Train error: ' + str(Pe_train[np.argmin(Pe_val)]))
print('Test error: ' + str(Pe_test[np.argmin(Pe_val)]))
print('Number of support vectors: ' + str(numSvs[np.argmin(Pe_val)]))

print('Pe_test: ', Pe_test)
print('Pe_val: ', Pe_val)
print('numSvs: ', numSvs)

display_SVs('rbf', 10, gamma=Gammaval_opt)
