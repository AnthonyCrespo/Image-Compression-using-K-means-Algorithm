#-------------------------------------- Analysis of Algorithms ---------------------------------------------------
#------------------------------------------- Final Project -------------------------------------------------------
#----------------------------------------- K-means Algorithm -----------------------------------------------------

# By: Brian Anthony Crespo Yaguana

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
import time
# ---------------------------------------- K-means ----------------------------------------

# -------------------------------- Centroids Initialization ---------------------------------

# 1. Random Initialization

def random_initialization(X, k):
    # k random rows of X are taken as the initial centroids.
    C = np.array([X[np.random.randint(X.shape[0])] for i in range(k)])
    return C

# 2. K-means ++  Initialization

def k_means_pp_initialization(X,k):
    
    # The first centroid is randomly selected 
    C = np.array([X[np.random.randint(X.shape[0])]])
    #print('Centroid 1', end =" ")
    # The following k-1 centroids are selected 
    for i in range(1,k):
        #print(f'Centroid {i+1}', end =" ")
        #distances stores the min distance between X[j] with regard to the centroids
        distances = np.zeros(X.shape[0])
        for j in range(X.shape[0]):
            #distances between X[j] and C
            d = np.linalg.norm(X[j] - C, axis = 1)
            # index_min: index of the min distance in d
            index_min = np.argmin(d) 
            # Store the min distance in distances
            distances[j] = d[index_min]
        # index_max: index of the row of max distance 
        index_max = np.argmax(distances)
        # X[j] with the maximum distance to a centroid is selected as the next centroid
        C = np.append(C, [ X[index_max] ], axis = 0)
    return C


#--------------------------- Assignment Step --------------------------------
def assignment_step(X,C):
    L = np.zeros(X.shape[0], dtype = int)
    for i in range(X.shape[0]):
        # Array of distances between X[i] and each centroid in C
        distances = np.linalg.norm(X[i] - C, axis = 1)
        #(X[i] - C) calculates all the distances between X[i] and the centroids
        #np.linalg.norm(X[i] - C, axis = 1) calculates norm 2 or euclidean distance for each row (axis = 1)
        
        # Index (number) of the centroid closest to X[i]      
        index = np.argmin(distances) 
        # index is assigned to the ith element of L, i.e, 
        # the number of the closest centroid is assigned to the element
        L[i] = index 
    return L

#-------------------------------- Update Step ----------------------------------
def update_step(X,L,C):
    for i in range(C.shape[0]):
        # Mean of the elements that belong to the ith cluster
        # L == i: True if the elements of L == i , else false
        C[i] = np.mean(X[ L == i ], axis = 0 )
    return C


# ----------------------- Stop Criterion 1 -----------------------------------
# ----------------------- Labels comparison -----------------------------------
def label_comparison(old_L, L):
    comparison = old_L == L
    return comparison.all()


# ----------------------- Stop Criterion 2 -----------------------------------
# ----------------------- Centroids distance -----------------------------------
def centroid_comparison(old_C, C):
    distances = np.linalg.norm(C - old_C, axis = 1)
    #print(distances)
    return np.max(distances)



# ----------------------- K-means Algorithm -----------------------------------

# L: Labels array (number of cluster for each row)
# C: Array with the centroids of each cluster
# k_means by default does 15 iterations
# k-means by default applies a random initialization of centroids
def k_means(X,k,iterations = 15, init = 'random',tol = 1e-3):

    # L contains as many labels as rows in X
    L = np.zeros(X.shape[0], dtype = int) 

    # Initialization of the centroids

    if init == 'random':
        C = random_initialization(X,k)
    elif init == 'k-means++':
        C = k_means_pp_initialization(X,k)
    #else: return 
    #---> Stop Criterion 3
    for i in range(iterations): 
        #print(f'Iteration {i+1}',end=" ")
        # Assignment Step
        old_L = np.copy(L)
        L = assignment_step(X,C)
        #---> Stop Criterion 2
        if label_comparison(old_L,L):
                print('There is no change in the centroids assignation.')
                break

        # Update Step
        old_C = np.copy(C)
        C = update_step(X,L,C)
        #---> Stop Criterion 1
        if centroid_comparison(old_C, C) < tol:
            print('Convergence has been reached!')
            break

    return C, L




# ---------------------------------------- Image Compression ----------------------------------------

# --------------------------------- Image-Array Conversion Functions --------------------------------


def image2array(image):
    image = np.asarray(image)
    image = image/255 # Data Normalization
    X=np.reshape(image,(image.shape[0]*image.shape[1],3))
    return X


def array2image(array,dimensions):
    array=array*255
    array= np.array(array, dtype = np.uint8)
    array = np.reshape(array,(dimensions[1], dimensions[0], 3))
    compressed = Image.fromarray(array)
    return compressed 



# -------------------------- Plotting Functions -------------------------------
plt.ioff()
def plot_comparison(original,compressed,k,im_name,init):
    fig, ax = plt.subplots(1, 2, figsize=(10, 10),
                        subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(wspace=0.05)
    ax[0].imshow(original)
    ax[0].set_title('Original Image', size=16)
    ax[1].imshow(compressed)
    ax[1].set_title(f'{k}-color Image', size=16);
    fig.set_size_inches(13, 6.5)
    fig.savefig(f'results/comparison_{im_name}_k={k}_{init}.png')


def plot_pixels(data,title,name,k,colors=None, N=10000000):
    if colors is None:
        colors = data
    
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))
    
    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    fig.labelsize = 12
    fig.suptitle(title, size=16);
    if 'Input' in title:
        fig.savefig(f'results/{name} {title}.png')
    else:
        fig.savefig(f'results/{name} {title} k={k}.png')

# ----------------------- Image Compression ------------------------------------
 
def image_compression(name,k,iterations = 15, initialization = 'random'):
    if not os.path.exists('results'):
        os.makedirs('results')
    
    print(f'Compressing {name} ...\n')
    
    image = Image.open(name)
    image = image.convert("RGB")
    X = image2array(image)
   

    
    # Unique colors in the image
    unique_rows = np.unique(X, axis=0)
        #-------------> Plot space color - Original Image
    plot_pixels(X,f'Input color space {unique_rows.shape[0]} colors',os.path.splitext(name)[0],k)
    
    
    
    #---------------------------------------------------------------------------
    #-----------------Apply K-means to the Image--------------------------------
    
    t = time.time()
    C, L = k_means(X,k,iterations,initialization)
    t = time.time() - t
    print(f"K-means execution time: {t} seconds.\n")
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    
    
    # Generate a 2D matrix, with the RGB values in the centroids 
    compressed = C[L]
    
    #-------------> Plot space color - Compressed Image
    plot_pixels(compressed,f'Reduced color space {k} colors', os.path.splitext(name)[0] , k )
    
    
    # Image is generated from the matrix
    compressed = array2image(compressed,image.size)
    
    # The image is saved
    only_name = os.path.splitext(name)[0] #Image Name
    extension = os.path.splitext(name)[1] #Image Extension
    compressed.save('results/'+only_name +'_comp_'+str(k)+'colors_'+str(initialization)+extension)
    
    print("Compression process finished.\n")
    print(f'Original number of colors: {unique_rows.shape[0]}')
    print(f'Number of colors after compression: {k}')
    
    #-------------> Plot space Image Comparison
    plot_comparison(image,compressed,k,only_name,initialization)
    #-------------------------------------------------

