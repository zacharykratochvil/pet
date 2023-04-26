import cv2
import annoy
import numpy as np
import sklearn.neighbors as skln
import sklearn.linear_model as sklm
import scipy as sp



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