import random
import sys

if len(sys.argv) != 2: # 3 because the first argument is the program name
    print("Usage: python Perceptron.py ionosphere.data")
    sys.exit()

file = sys.argv[1]
instances = []

#load the data
with open(file, 'r') as f:
    next(f)
    for line in f:
        instances.append([attribute for attribute in line.split(' ')])
for instance in instances:
    for i in range(len(instance)-1):
        instance[i] = float(instance[i])
    instance[-1] = instance[-1].strip('\n') # remove the newline character from the class label

# weights = [0 for i in range(len(instances[0]) - 1)]
# threshold = 0 #initialise threshold to 0

#initialise weights and threshold to random values
weights = [random.uniform(-1, 1) for i in range(len(instances[0]) - 1)]
threshold = random.uniform(-1, 1)

# train the perceptron
def trainPerceptron(trainingInstances):
    global threshold
    global weights
    countCorrect = 0
    count = 0
    lastCorrect = 0 #number of correctly classified instances in the previous iteration
    maxIterations = 5000 #maximum number of iterations allowed
    convergence = 0
    while countCorrect < len(trainingInstances) and count < maxIterations:
        countCorrect = 0
        count += 1

        for instance in trainingInstances:
            output = sum([weights[i]*instance[i] for i in range(len(weights))]) + threshold
            #if the output is not the same as the class label, update the weights and threshold
            prediction = 'b' if output < 0 else 'g'
            if prediction == 'b' and instance[-1] == 'g': #prediction too low
                for i in range(len(weights)):
                    weights[i] += instance[i]
                threshold += 1
            elif prediction == 'g' and instance[-1] == 'b': #prediction too high
                for i in range(len(weights)):
                    weights[i] -= instance[i]
                threshold -= 1
            else:
                countCorrect += 1
        
        if countCorrect == lastCorrect: #no improvements this iteration
            convergence += 1
        
        if convergence >= 100: #no improvements for 100 iterations, stop
            break

        lastCorrect = countCorrect

    return count

def correctPredictions(testInstances):
    global threshold
    global weights
    correct = 0
    for instance in testInstances:
        output = sum([weights[i]*instance[i] for i in range(len(weights))]) + threshold
        prediction = 'b' if output < 0 else 'g'
        if prediction == instance[-1]:
            correct += 1
    return correct

#report results
print("PERCEPTRON:")
print("Number of iterations (maximum 5000): "+str(trainPerceptron(instances)))
print("Number of correctly classified instances: "+str(correctPredictions(instances))+ " out of "+str(len(instances))+" instances.")
print("Number of incorrectly classified instances: "+str(len(instances)-correctPredictions(instances))+" out of "+str(len(instances))+" instances.")
print("Classification accuracy: {:.1f}%".format(correctPredictions(instances)/len(instances)*100))
print("Weights: "+str(weights))
print("Threshold (w0): "+str(threshold))
print("--------------------------------------------------")
print("SPLITTING THE DATA INTO TRAINING AND TEST SETS:")

#split the data into training and test sets
sliceIndex = round(len(instances) * 0.7)
trainingInstances = instances[:sliceIndex]
testInstances = instances[sliceIndex:]
weights = [random.uniform(-1, 1) for i in range(len(instances[0]) - 1)]
threshold = random.uniform(-1, 1)
print("Number of training instances: "+str(len(trainingInstances)))
print("Number of test instances: "+str(len(testInstances)))

print("Number of iterations (maximum 5000): "+str(trainPerceptron(trainingInstances)))
print("Number of correctly classified instances: "+str(correctPredictions(testInstances))+ " out of "+str(len(testInstances))+" instances.")
print("Number of incorrectly classified instances: "+str(len(testInstances)-correctPredictions(testInstances))+" out of "+str(len(testInstances))+" instances.")
print("Classification accuracy: {:.1f}%".format(correctPredictions(testInstances)/len(testInstances)*100))
print("Weights: "+str(weights))
print("Threshold (w0): "+str(threshold))




