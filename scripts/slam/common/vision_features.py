import cv2
import annoy
import numpy as np
import sklearn.neighbors as skln
import sklearn.linear_model as sklm
import scipy as sp
import rospy
import time


## from https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774
# Using KAZE, cause SIFT, ORB and other was moved to additional module
# which is adding addtional pain during install
def detect(image, num_points = 32, descriptors_per_point = 64, how="SIFT"):
    if how == "SIFT":
        alg = cv2.SIFT_create()
    elif how == "KAZE":
        alg = cv2.KAZE_create()

    # Finding image keypoints
    kps = alg.detect(image)


    # Getting first 32 of them. 
    # Number of keypoints is varies depend on image size and color pallet
    # Sorting them based on keypoint response value(bigger is better)
    kps = sorted(kps, key=lambda x: -x.response)[:num_points]

    # computing descriptors vector
    kps, dsc = alg.compute(image, kps)

    pts = np.array([k.pt for k in kps])

    # Making descriptor of same size
    # Descriptor vector size is 64
    if len(pts) == 0:
        actual_point_count = 0
    else:
        actual_point_count = min(num_points, np.shape(pts)[0])
    final_dsc = np.zeros([actual_point_count, descriptors_per_point])
    
    if type(dsc) is not type(None):
        for i in range(actual_point_count):
            len_to_copy = min(descriptors_per_point, len(dsc[i]))
            final_dsc[i,:len_to_copy] = dsc[i,:len_to_copy]

    return pts, final_dsc


def compare_images(base_coords, compare_coords, base_features, compare_features):
    # fit tree for pixel coordinates in base image
    nn_tree = skln.NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree",
                                            leaf_size = 10, n_jobs = 4)
    nn_tree.fit(base_coords)

    # compare base features to compare features
    feature_distance = 0
    feature_sum = 0
    for compare_i in range(len(compare_coords)):
        
        # find base points near this compare point
        neighbors = nn_tree.radius_neighbors([compare_coords[compare_i]], radius = 20, return_distance = True)
        for neighbor_i in range(np.shape(neighbors[0][0])[0]):

            base_i = neighbors[1][0][neighbor_i]
            #coord_distance = neighbors[0][0][neighbor_i]

            base_feature_vector = base_features[base_i]
            compare_feature_vector = compare_features[compare_i]
            feature_sum += np.sum([base_feature_vector, compare_feature_vector])
            feature_distance += sp.spatial.distance.cityblock(base_feature_vector,compare_feature_vector)

    # return image similarity
    if feature_sum == 0:
        return 0
    else:
        return 1-feature_distance/feature_sum

def estimate_translation_and_zoom(base_feature_coords, translated_feature_coords, base_features, translated_features, image_shape):
    if len(translated_feature_coords) == 0:
        return 0, 0, np.inf
    
    # fit tree for base features
    #nn_tree = skln.NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree",
    #                                        leaf_size = 5, n_jobs = 1)
    #nn_tree.fit(base_features)
    f = base_features.shape[1]
    t = 50
    ai = annoy.AnnoyIndex(f, metric='euclidean')
    for i in range(base_features.shape[0]):
        ai.add_item(i, base_features[i,:])

    ai.build(t, n_jobs=1)

    # calculate most similar translated features and their translation
    num_features = np.shape(translated_feature_coords)[0]
    translations = np.empty([num_features, 2])
    scores = np.empty([num_features])
    for i in range(num_features):
        
        neighbors_to_find = 1
        neighbor = ai.get_nns_by_vector(translated_features[i,:], neighbors_to_find, search_k=-1, include_distances=True)
        local_scores = np.array(neighbor[1])
        #neighbor = nn_tree.kneighbors([translated_features[i,:]], neighbors_to_find, return_distance=True)
        #local_scores = neighbor[0].flatten()
        
        average_base_coord = np.average(base_feature_coords[neighbor[0],:], weights=np.exp(-local_scores)+.001, axis=0)
        #average_base_coord = np.average(base_feature_coords[neighbor[1],:], weights=np.exp(-local_scores)+.001, axis=1)
        translations[i, :] = translated_feature_coords[i,:] - average_base_coord
        scores[i] = np.min(local_scores)

    # calculate average translation, zoom, and score
    x_model = sklm.LinearRegression()
    xx = translated_feature_coords[:,:1]
    xy = translations[:,0].flatten()
    weights = np.exp(-scores).flatten()
    x_model.fit(xx, xy, weights)
    zoom_x = x_model.coef_[0]
    translate_x = x_model.predict(np.reshape(image_shape[1]/2,[-1,1]))#x_model.intercept_

    y_model = sklm.LinearRegression()
    yx = translated_feature_coords[:,1:]
    yy = translations[:,1].flatten()
    y_model.fit(yx, yy, weights)
    zoom_y = y_model.coef_[0]
    translate_y = y_model.predict(np.reshape(image_shape[0]/2,[-1,1]))#y_model.intercept_

    zoom = zoom = 1/(np.mean([zoom_x, zoom_y])+1)
    translation = -np.array((translate_x.squeeze(), translate_y.squeeze()))
    score = np.mean([x_model.score(xx, xy, weights), y_model.score(yx, yy, weights)])
    #score = np.mean(scores)

    # return
    return translation, zoom, score

# based on https://openaccess.thecvf.com/content/CVPR2021/papers/Jang_MeanShift_Extremely_Fast_Mode-Seeking_With_Applications_to_Segmentation_and_Object_CVPR_2021_paper.pdf
def meanshift(X, bandwidth, tolerance, max_iter, kernel=lambda x: x**2 < 1):
    y = np.zeros([max_iter, *np.shape(X)])
    y[0,:,:] = X

    t = 1
    while np.sum((y[t,:,:] - y[t-1,:,:])**2) > tolerance and t < max_iter:
        for i in range(np.shape(X)[0]):

            numerator_sum = 0
            for j in range(np.shape(X)[0]):
                numerator_sum += kernel((y[t-1,i,:] - y[t-1,j,:])**2/bandwidth)*y[t-1,j,:]

            denominator_sum = 0
            for j in range(np.shape(X)[0]):
                denominator_sum += kernel((y[t-1,i,:] - y[t-1,j,:])**2/bandwidth)
            
            y[t,:,:] = numerator_sum/denominator_sum

        t += 1

    return np.squeeze(y[t,:,:])

def meanshift_plus(X, bandwidth, tolerance, max_iter):
    '''
    Calculates meanshift++ from paper.
    '''

    y = np.zeros([max_iter, *np.shape(X)],dtype="float16")
    y[0,:,:] = X
    num_samples = np.shape(X)[0]
    adjacent_cells = (np.asarray((-1,-1)),np.asarray((-1,0)),np.asarray((-1,1)),np.asarray((0,-1)),np.asarray((0,0)),np.asarray((0,1)),np.asarray((1,-1)),np.asarray((1,0)),np.asarray((1,1)))
    
    t = 1
    while t < max_iter and np.sum((y[t,:,:] - y[t-1,:,:])**2) > tolerance:
        
        # initialize hash tables with current counts and sums for each grid cell
        counts = {}
        sums = {}
        for s in range(num_samples):
            grid_key = tuple(np.asarray(y[t-1,s,:]/bandwidth,int))

            if grid_key not in counts:
                counts[grid_key] = 1
                sums[grid_key] = y[t-1,s,:].copy()
            else:
                counts[grid_key] += 1
                sums[grid_key] += y[t-1,s,:].copy()

        # compute new mean of each cell based on adjacent cells
        for s in range(num_samples):
            grid_key = tuple(np.asarray(y[t-1,s,:]/bandwidth,int))

            numerator_sum = np.zeros(np.shape(X)[1])
            denominator_sum = 0
            for v in adjacent_cells:
                adj_key = (grid_key[0] + v[0], grid_key[1] + v[1])

                if adj_key in sums:
                    numerator_sum += sums[adj_key]
                    denominator_sum += counts[adj_key]

            y[t,s,:] = numerator_sum/float(denominator_sum)

        t += 1

    return np.squeeze(y[t-1,:,:])

# From https://stackoverflow.com/questions/23494037/how-do-you-calculate-the-median-of-a-set-of-angles#:~:text=To%20get%20the%20median%20of,have%20more%20than%20one%20angle.
def angle_interpol(a1, w1, a2, w2):
    """Weighted avarage of two angles a1, a2 with weights w1, w2"""

    diff = a2 - a1        
    if diff > 180: a1 += 360
    elif diff < -180: a1 -= 360

    aa = (w1 * a1 + w2 * a2) / (w1 + w2)

    if aa > 360: aa -= 360
    elif aa < 0: aa += 360

    return aa

def angle_mean(angle):    
    """Unweighted average of a list of angles"""

    aa = 0.0
    ww = 0.0

    for a in angle:
        aa = angle_interpol(aa, ww, a, 1)
        ww += 1

    return ((aa + 180) % 360) -180

'''
def estimate_translation_and_zoom(base_feature_coords, translated_feature_coords, base_features, translated_features, image_shape):
    if len(translated_feature_coords) == 0:
        return 0, 0, np.inf
    
    # fit tree for base features
    nn_tree = skln.NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree",
                                            leaf_size = 5, n_jobs = 4)
    nn_tree.fit(base_features)

    # calculate most similar translated features and their translation
    num_features = np.shape(translated_feature_coords)[0]
    translations = np.empty([num_features, 2])
    scores = np.empty([num_features])
    for i in range(num_features):
        
        neighbors_to_find = 1
        neighbor = nn_tree.kneighbors([translated_features[i,:]], neighbors_to_find, return_distance=True)
        local_scores = neighbor[0].flatten()
        average_base_coord = np.average(base_feature_coords[neighbor[1],:], weights=np.exp(-local_scores)+.001, axis=1)
        translations[i, :] = translated_feature_coords[i,:] - average_base_coord
        scores[i] = np.min(local_scores)

    # calculate average translation, zoom, and score
    x_model = sklm.LinearRegression()
    xx = translated_feature_coords[:,:1]
    xy = translations[:,0].flatten()
    weights = np.exp(-scores).flatten()
    x_model.fit(xx, xy, weights)
    zoom_x = x_model.coef_[0]
    translate_x = x_model.predict(np.reshape(image_shape[1]/2,[-1,1]))

    y_model = sklm.LinearRegression()
    yx = translated_feature_coords[:,1:]
    yy = translations[:,1].flatten()
    y_model.fit(yx, yy, weights)
    zoom_y = y_model.coef_[0]
    translate_y = y_model.predict(np.reshape(image_shape[0]/2,[-1,1]))

    zoom = zoom = 1/(np.mean([zoom_x, zoom_y])+1) 
    translation = -np.array((translate_x.squeeze(), translate_y.squeeze()))
    score = np.mean([x_model.score(xx, xy, weights), y_model.score(yx, yy, weights)])

    # return
    return translation, zoom, score
'''