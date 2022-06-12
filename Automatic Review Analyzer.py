from string import punctuation, digits
import numpy as np
import random
import csv
import sys

if sys.version_info[0] < 3:
    PYTHON3 = False
else:
    PYTHON3 = True

# Part I


def get_order(n_samples):
    try:
        with open("C:\\sentiment_analysis\\" + str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices



def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    # Your code here
    Loss = label*(np.matmul(theta, feature_vector) + theta_0)
    hinge = np.max([0.0, 1-Loss])
    return hinge
    raise NotImplementedError


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    
    count = 0
    num = len(feature_matrix)
    
    for i in range(num):
        count = count + hinge_loss_single(feature_matrix[i], labels[i], theta, theta_0)
   
    return count/num
    raise NotImplementedError


def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    
    if label*(np.matmul(current_theta, feature_vector) + current_theta_0) <= 1e-14:

        current_theta = current_theta + label*feature_vector
        current_theta_0 = current_theta_0 + label
    
    return current_theta, current_theta_0
    raise NotImplementedError


def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    theta = np.zeros((feature_matrix.shape[1])) 
    theta_0 = 0.0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
    
    return theta, theta_0
   


def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])


    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    
    theta_sum = np.zeros((feature_matrix.shape[1]))
    theta_o = 0.0
    theta = np.zeros((feature_matrix.shape[1])) #make theta dynamic
    theta_0 = 0.0
    n = feature_matrix.shape[0] #can also be len(feature_matrix)
    
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
            theta_sum += theta
            theta_o += theta_0
    
    return theta_sum/(n*T), theta_o/(n*T)
    

    raise NotImplementedError


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    
    m = 1 - L*eta
    if label*(np.matmul(current_theta, feature_vector) + current_theta_0) <= 1:
        current_theta = m* current_theta + eta*label*feature_vector
        current_theta_0 = current_theta_0 + eta*label
        return current_theta, current_theta_0
    else:
        current_theta = m* current_theta
        return current_theta, current_theta_0
    raise NotImplementedError


def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """

    current_theta = np.zeros(feature_matrix.shape[1])
    current_theta_0 = 0
    count = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]): 
            count +=1
            eta = 1.0/np.sqrt(count)
            current_theta, current_theta_0 = pegasos_single_step_update(feature_matrix[i],
            labels[i],L,eta,current_theta,current_theta_0)

    return current_theta, current_theta_0
    raise NotImplementedError

# Part II


def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
   
    n = feature_matrix.shape[0]
    classification = np.zeros(n)
    def onestep_classifier(vector):
        if np.matmul(theta, vector) + theta_0 >= 1e-16:
            return 1
        else:
            return -1
    
    for i in range(n):
        classification[i] = onestep_classifier(feature_matrix[i])


    return classification
    raise NotImplementedError
feature_matrix = np.array([[1, 1], [1, 1], [1, 1]])
theta = np.array([1, 1])
theta_0 = 0

a = classify(np.array([[1, 1], [1, 1], [1, 1]]), np.array([1, 1]), 0)


def accuracy(preds, targets):
	"""
	Given length-N vectors containing predicted and target labels,
	returns the percentage and number of correct predictions.
	"""
	return (preds == targets).mean()

def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.
    The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the validation
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """
    
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    

    clss_train = classify(train_feature_matrix, theta, theta_0)
    clss_val = classify(val_feature_matrix, theta, theta_0)

    acc_train = accuracy(clss_train, train_labels)
    acc_val = accuracy(clss_val, val_labels)

    return acc_train, acc_val


    raise NotImplementedError


def load_data(path_data, extras=False):
    """
    Returns a list of dict with keys:
    * sentiment: +1 or -1 if the review was positive or negative, respectively
    * text: the text of the review

    Additionally, if the `extras` argument is True, each dict will also include the
    following information:
    * productId: a string that uniquely identifies each product
    * userId: a string that uniquely identifies each user
    * summary: the title of the review
    * helpfulY: the number of users who thought this review was helpful
    * helpfulN: the number of users who thought this review was NOT helpful
    """

    global PYTHON3

    basic_fields = {'sentiment', 'text'}
    numeric_fields = {'sentiment', 'helpfulY', 'helpfulN'}

    data = []
    if PYTHON3:
        f_data = open(path_data, encoding="latin1")
    else:
        f_data = open(path_data)

    for datum in csv.DictReader(f_data, delimiter='\t'):
        for field in list(datum.keys()):
            if not extras and field not in basic_fields:
                del datum[field]
            elif field in numeric_fields and datum[field]:
                datum[field] = int(datum[field])

        data.append(datum)

    f_data.close()

    return data


def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()


def getting_text(dataset):
    txts = []

    for d in dataset:
        for key, value in d.items():
            if key == 'sentiment':
                continue
            else:
                txts.append(value)
    return txts

def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    
    def stop_words(dataset):

         txt_file = open(dataset, 'r')
         data = txt_file.read()
         to_list = data.split('\n')
         txt_file.close()
         return to_list
         
    
    dictionary = {} # maps word to unique index
    call_stop_words = stop_words("C:\\sentiment_analysis\\stopwords.txt")
    for text in texts:
        word_list = extract_words(text)
        #update to original code to remove stop words


        for word in word_list:
            if word not in dictionary and word not in call_stop_words:
                dictionary[word] = len(dictionary)
    return dictionary



def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9
    """

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = feature_matrix[i, dictionary[word]] + 1
    return feature_matrix


feature_x = extract_bow_feature_vectors(getting_text(load_data("C:\\sentiment_analysis\\reviews_train.tsv")), bag_of_words(getting_text(load_data("C:\\sentiment_analysis\\reviews_train.tsv"))))
#print(feature_x)


def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
