import random
import sys

if len(sys.argv) != 3: # 3 because the first argument is the program name
    print("Usage: python Wine.py wine-training wine-test")
    sys.exit()

training = sys.argv[1]
test = sys.argv[2]

trainingData = []
testData = []
k = 3 # change depending on the k value

# Read the training file and extract the class labels
with open(training, 'r') as t:
    next(t)
    for line in t:
        # Convert the attributes to floats and append them to trainingData
        trainingData.append([float(attribute) for attribute in line.split(' ')]) 

# Read the test file and extract the class labels
with open(test, 'r') as t:
    next(t)
    for line in t:
        # Convert the attributes to floats and append them to testData
        testData.append([float(attribute) for attribute in line.split(' ')])

def findRange(i):
    column = [row[i] for row in trainingData] # get the entire column of the i'th attribute
    return max(column) - min(column)

def findDistance(testPoint, trainingPoint):
    distance = 0
    for i in range(len(testPoint) - 1):
        distance += (((testPoint[i] - trainingPoint[i]) ** 2) / (findRange(i) ** 2))
    return distance ** 0.5

def findNearestNeighbors(testPoint):
    distances = []
    for i in range(len(trainingData)):
        # append the distance and the class label to distances
        distances.append([findDistance(testPoint, trainingData[i]), trainingData[i][-1]]) 
    distances.sort(key=lambda x: x[0]) # sort the distances by the first element (the distance)
    return distances[:k]

# classify each test point
classifiers = []
for testPoint in testData:
    nearestNeighbors = findNearestNeighbors(testPoint)
    classLabels = [neighbor[1] for neighbor in nearestNeighbors] # get the class labels
    classifiers.append(round(max(set(classLabels), key=classLabels.count))) # append the most common class label


print("when k = " + str(k) + ":")
print("Predictions for test data: " + repr(classifiers)) # print the predicted class labels
count = 0
for i in range(len(classifiers)):
    if classifiers[i] == testData[i][-1]:
        count += 1
print("Accuracy: {:.1f}% ".format( (count / len(classifiers)*100))) # print the accuracy rate

def kFoldCrossValidation(fold, trainingData, testData):
    # split the data into k folds
    folds = []
    totalData = trainingData + testData
    #split the data into k folds
    for i in range(fold):
        folds.append(totalData[i::fold])
    # shuffle the data
    random.shuffle(folds)

    accuracies = []
    for i in range(fold):
        # get the test data
        testData = folds[i]
        # get the training data
        trainingData = [row for fold in folds if fold != testData for row in fold]
        # classify each test point
        classifiers = []
        for testPoint in testData:
            nearestNeighbors = findNearestNeighbors(testPoint)
            classLabels = [neighbor[1] for neighbor in nearestNeighbors] # get the class labels
            classifiers.append(round(max(set(classLabels), key=classLabels.count))) # append the most common class label
        count = 0
        for i in range(len(classifiers)):
            if classifiers[i] == testData[i][-1]:
                count += 1
        accuracies.append(count / len(classifiers))
    return sum(accuracies) / len(accuracies)

#print(kFoldCrossValidation(5, trainingData, testData))
print("Accuracy with 5-fold cross validation: {:.1f}% ".format(kFoldCrossValidation(5, trainingData, testData) * 100))
