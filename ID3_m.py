import sys
from scipy import stats
import pickle as pkl

def main():
    '''
        Usage: ID3_m.py -p <pvalue> -f1 <train_dataset> -f2 <test_dataset> -o <output_file> -t <decision_tree>
            -p      The p-value to use for statistical relevance tests
            -f1     The csv formatted training dataset
            -f2     The csv formatted test data set
            -o      The path/name of the output file
            -t      The decision tree output path

        This function creates a linked set of nodes that process a list of features and returns a 'will click',
        'won't click' or 'unkown' prediciton
    '''
    import sys

    # format training data
    trainingData = open(sys.argv[4], "r")
    trainingData = trainingData.readlines()
    parsedTrainingData = []
    for i in range(len(trainingData)):
        parsedTrainingData.append(trainingData[i].split())

    trainingLabelPath = sys.argv[4].split('.')[0] + '_label.csv'
    trainingLabels = open(trainingLabelPath, "r")
    labelList = trainingLabels.readlines()

    testPValue = float(sys.argv[2])

    # generate decision tree using test data
    decisionTree = ID3Ply(parsedTrainingData, labelList, testPValue, [])
    
    # translate decision tree into a generic tree
    #outputTree = DecisionTreetoGenericTree(decisionTree)
    # save tree to output
    #outputTree.save_tree(sys.argv[10])

    print("tree generated")
    # format test data
    testData = open(sys.argv[6], "r")
    testData = testData.readlines()
    parsedTestData = []
    for i in range(len(testData)):
        parsedTestData.append(testData[i].split())

    testLabelPath = sys.argv[6].split('.')[0] + '_label.csv'
    testLabels = open(testLabelPath, "r")
    labelList = testLabels.readlines()
    
    #output = open(sys.argv[8], "w")

    # evaluate test data
    hit = 0
    miss = 0
    unkown = 0
    for i in range(len(testData)):
        #print(parsedTestData[i])
        guess = decisionTree.evaluate(parsedTestData[i])

        #print("guessing:", guess)
        if guess in labelList[i]:
            hit += 1
        else:
            miss += 1
        
        #write output to file
        #output.write(str(guess) + "\n")
    
    # print results
    accuracy = float(hit) / float((hit + miss + unkown))
    print("predicted", hit, "hits", miss, "misses", unkown, "unkown", accuracy, "accuracy rate")
    

def ID3Ply(trainingData, labelList, testPValue, skipList):

    # create feature list if first loop
    featureList = list()
    for i in range(len(trainingData[0])):
        featureList.append(Feature(i))

    # parse training data into data object clases
    labelIndex = 0
    posTotal = 0
    for line in trainingData:
        # update the hit/miss count of each feature
        for i in range(len(line)):
            featureList[i].updateAttr(line[i], int(labelList[labelIndex]))
        # track total pos/negative labels
        if "1" in labelList[labelIndex]:
            posTotal += 1
        labelIndex += 1
    negTotal = labelIndex - posTotal

    # evaluate features and build next node of tree
    print("ID3 eval features")
    for i in range(len(featureList)):
        if i in skipList:
            continue

        feature = featureList[i]
        actual = []
        expected = []
        for attr in feature.getAttrs():

            # calculate expected value
            ePos = float(feature.occCount(attr) * posTotal/labelIndex)
            eNeg = float(feature.occCount(attr) * negTotal/labelIndex)

            # set expected and actual values
            if ePos != 0:
                expected.append(ePos)
                actual.append(feature.posCount(attr))
            if eNeg != 0:
                expected.append(eNeg)
                actual.append(feature.negCount(attr))

        degfreedom = len(feature.getAttrs()) - 1
        chi2, curPValue = stats.chisquare(actual, f_exp = expected, ddof = degfreedom)

        # if feature has passed relevance test, use as next decision node
        if curPValue <= testPValue:
            print("ID3 create node on feature", i, feature.attributes)
            # create decision node
            newNode = DecisionNode(i + 1, round(feature.posTotal()/feature.occTotal()) )
            skipList.append(i)

            for attr in feature.getAttrs():
                
                # build new data set
                newTrainingData = []
                newLabelList = []
                    
                # only add test cases where the feature value matches attr value

                trainTrue = 0
                for j in range(len(labelList)):
                    if "1" in labelList[j]:
                        trainTrue += 1
                    line = trainingData[j]
                    if line[i] == attr:
                        newTrainingData.append(trainingData[j])
                        newLabelList.append(labelList[j])
                        
                # leaf node case - no training cases
                if len(newTrainingData) == 0:
                    print("leaf node no more cases")
                    finVal = round(trainTrue/len(labelList))
                    print("training true:", trainTrue, "len: ", len(labelList), "finval: ", finVal)
                    newNode.addBranch(attr, LeafNode(finVal))
                # leaf node case: no features
                elif len(newTrainingData[0]) == 0:
                    print("leaf node no more features")
                    totTrue = 0
                    for label in newLabelList:
                        if "1" in label:
                            totTrue += 1
                    finVal = round(totTrue/len(newLabelList))
                    print("no features at all:", finVal)
                    newNode.addBranch(attr, LeafNode(finVal))
                # add attr value branch to feature node
                else:
                    print("decision node recursing")
                    newNode.addBranch(attr, ID3Ply(newTrainingData, newLabelList, testPValue, skipList))
            return newNode

    # leaf node case - no features meet p-value signifigance test
    finVal = round(posTotal/labelIndex)
    print("leaf node no significant features", finVal)
    return LeafNode(finVal)

class Feature():
    def __init__(self, id):
        self.attributes = dict()
        self.id = id
        self.posTot = 0
        self.occTot = 0

    def getID(self):
        return self.id

    def getAttrs(self):
        return self.attributes.keys()

    def isAttr(self, attr):
        return self.attributes.has_key(attr)

    def updateAttr(self, attr, posOrNeg):
        self.occTot += 1
        self.posTot += posOrNeg

        pos = posOrNeg
        neg = 1 - posOrNeg
        occ = 1

        if self.isAttr(attr):
            oldPos = self.attributes[attr][0]
            oldNeg = self.attributes[attr][1]
            oldOcc = self.attributes[attr][2]
            self.attributes[attr] = (pos + oldPos, neg + oldNeg, occ + oldOcc)
        else:
            self.attributes[attr] = (pos, neg, occ)

    def posCount(self, attr):
        return self.attributes[attr][0]

    def negCount(self, attr):
        return self.attributes[attr][1]

    def occCount(self, attr):
        return self.attributes[attr][2]

    def posTotal(self):
        return self.posTot

    def occTotal(self):
        return self.occTot

class DecisionNode():
    def __init__(self, id, defaultval):
            self.id = id
            self.checks = []
            self.defaultval = defaultval

    def addBranch(self, attr, node):
        self.checks.append((attr, node))

    def evaluate(self, featureVals):
        for check in self.checks:
            if featureVals[int(self.id) - 1] == check[0]:
                #print("passing")
                return check[1].evaluate(featureVals)
        return str(int(self.defaultval))

class LeafNode():
    def __init__(self, val):
        self.val = str(int(val))

    def evaluate(self, featureVals = None):
        #print("leaf")
        return self.val

def DecisionTreetoGenericTree(node):
    
    # translate leaf nodes
    if isinstance(node, LeafNode):
        data = 'T'
        if node.evaluate() == '0':
            data = 'F'
        return TreeNode(data)
    
    # translate decision nodes
    children = [-1, -1, -1, -1, -1]
    data = 'F'
    if node.defaultval > 0:
        data = 'T'
    for i in range(5):
        children[i] = TreeNode(data)

    for tuple in node.checks:
        featureVal = int(tuple[0]) - 1
        children[featureVal] = DecisionTreetoGenericTree(tuple[1])
    return TreeNode(node.id, children)

'''
    TreeNode represents a node in a decision tree
    TreeNode can be:
        - A non-leaf node: 
            - data: contains the feature number this node is using to split the data
            - children[0]-children[4]: Each correspond to one of the values that the feature can take
        
        - A leaf node:
            - data: 'T' or 'F' 
            - children[0]-children[4]: Does not matter, you can leave them the same or cast to None.
'''

class TreeNode():
    def __init__(self, data='T',children=[-1]*5):
        self.nodes = list(children)
        self.data = data
        
    def save_tree(self,filename):
        obj = open(filename,'w')
        pkl.dump(self,obj)


if __name__== "__main__":
        main()