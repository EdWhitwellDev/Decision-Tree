import pandas as pd
import numpy as np
import time

from statistics import mode

pd.options.mode.chained_assignment = None  # default='warn'

# This is the Tree i am currently working on
# The general idea is, instead of splitting the data by every category every time, i only split the data by the categories that significantly improve the accuracy of the prediction
# catagories that are not improved by splitting by that attribute are passed to a different node that will try to improve them without splitting up the data by that attribute first
# they basically side step that split, however the next node selectes its split attribute based on its ability to predict the previously unimproved catagories.

# at the moment the model gets 79.2% accuracy which is my best yet (with ML), and it does that without a forest, i am hoping to get it to 80% with only one tree by some point

class Tree:
    def __init__(self, TrainData, MaxDepth = 2, SolidThresh = 0.83, MinSize = 2):
        self.Attributes = TrainData.columns.drop(['Survived', 'Train', 'PassengerId'])
        self.MaxDepth, self.MinSize, self.SolidThresh = MaxDepth, MinSize, SolidThresh

        self.Data = TrainData
        self.DataSize = len(TrainData)

        self.GeneralPrediction = TrainData['Survived'].value_counts(normalize = True).idxmax()

        self.Root = ImprovingNode(self, 0, TrainData, TrainData)
        print("Tree grown")

    def Predict(self, Passenger):
        return round(self.Root.Predict(Passenger))

class ImprovingNode:
    def __init__(self, Tree, Depth, Data, MeasuredData, MeasuredAttribute = None, MeasuredValues = None):
        Stats = Data['Survived'].value_counts(normalize = True)
        self.Benchmark = Stats.max()
        self.Prediction = Stats.idxmax()

        self.SplitAttribute = None

        self.Tree, self.Depth = Tree, Depth

        if Depth == Tree.MaxDepth:
            return
        self.MeasuredAttribute, self.MeasuredValues, self.MeasuredData = MeasuredAttribute, MeasuredValues, MeasuredData

        self.PotentialSplitAttributes = self.Tree.Attributes
        if self.MeasuredAttribute != None: self.PotentialSplitAttributes = self.PotentialSplitAttributes.drop(self.MeasuredAttribute)

        self.SplitAttribute, ImprovedValues, NotImprovedValues, ImprovedAccuracies, ImprovedPredictions = self.FindSplit(Data)

        if self.SplitAttribute == None:
            return
        
        self.ImprovingNodes, self.ToImproveCats, self.ToSmall, self.Solid, self.SolidPredictions = [], [], [], [], []

        for Index in range(len(ImprovedValues)):
            if ImprovedAccuracies[Index] > self.Tree.SolidThresh:
                self.Solid.append(ImprovedValues[Index])
                self.SolidPredictions.append(ImprovedPredictions[Index])
            else:
                WouldBeUsedToPredict = MeasuredData.loc[MeasuredData[self.SplitAttribute] == ImprovedValues[Index]]
                DataToPredict = Data.loc[Data[self.SplitAttribute] == ImprovedValues[Index]]
                if len(WouldBeUsedToPredict) >= self.Tree.MinSize:
                    self.ImprovingNodes.append(ImprovingNode(self.Tree, self.Depth+1, DataToPredict, WouldBeUsedToPredict, self.SplitAttribute, ImprovedValues[Index]))
                    self.ToImproveCats.append(ImprovedValues[Index])
        
        for Index in range(len(NotImprovedValues)):
            ToBeMeasured = MeasuredData.loc[MeasuredData[self.SplitAttribute] == NotImprovedValues[Index]]
            if len(ToBeMeasured) >= self.Tree.MinSize:
                self.ImprovingNodes.append(ImprovingNode(self.Tree, self.Depth+1, Data, ToBeMeasured, self.SplitAttribute, NotImprovedValues[Index]))
                self.ToImproveCats.append(NotImprovedValues[Index])
            else: self.ToSmall.append(NotImprovedValues[Index])
                
    def Predict(self, Passenger):
        if self.SplitAttribute != None:
            Catagory = Passenger[self.SplitAttribute]
            for Index in range(len(self.Solid)):
                if Catagory == self.Solid[Index]:return self.SolidPredictions[Index]
                    
            for Index in range(len(self.ToImproveCats)):
                if Catagory == self.ToImproveCats[Index]:return self.ImprovingNodes[Index].Predict(Passenger)
            
            for Index in range(len(self.ToSmall)):
                if Catagory == self.ToSmall[Index]:return self.Tree.GeneralPrediction

        return self.Prediction
        
    def FindSplit(self, Data):
        BestAttribute, ImprovedValues, WorseValues, ImprovedAccuracies, ImprovedPredictions, BestAccuracy = None, None, None, None, None, 0, 

        MeasuredDataSize = len(self.MeasuredData)
        
        for Attribute in self.PotentialSplitAttributes:
            WholeGroupedPredictor = Data.groupby([Attribute])['Survived'].value_counts().unstack().idxmax(axis = 1)

            TempPrediction = self.MeasuredData[['Survived', Attribute]]
            TempPrediction['Prediction'] = TempPrediction[Attribute].map(WholeGroupedPredictor)
            TempPrediction['Accuracy'] = TempPrediction['Survived'] == TempPrediction['Prediction']

            CatGroupSize = TempPrediction.groupby([Attribute]).size()
            CatGroupProportion = CatGroupSize / MeasuredDataSize

            CorrectGroupSize = TempPrediction.loc[TempPrediction['Accuracy'] == True].groupby([Attribute]).size()
            CorrectPercentage = CorrectGroupSize / CatGroupSize

            BenchMask = CorrectPercentage >= min(self.Benchmark * 1.27, self.Tree.SolidThresh)
            RelevantPercetages = CorrectPercentage.loc[BenchMask]
            RelevantProportions = CatGroupProportion.loc[BenchMask]

            Adjusted = (RelevantPercetages) * (RelevantProportions) 

            Score = Adjusted.sum()
            if Score > BestAccuracy:
                BestAccuracy, BestAttribute, ImprovedValues, WorseValues, ImprovedAccuracies, ImprovedPredictions = Score, Attribute, Adjusted.index, CorrectPercentage.loc[~BenchMask].index, RelevantPercetages.values, WholeGroupedPredictor[Adjusted.index].values

        return BestAttribute, ImprovedValues, WorseValues, ImprovedAccuracies, ImprovedPredictions

def LoadData():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    test['Survived'] = None
    train['Train'] = True
    test['Train'] = False

    Full = pd.concat([train, test], sort = False)

    Map = {
        'Mr': 'Mr',
        'Mrs': 'Mrs',
        'Miss': 'Miss',
        'Master': 'Master',
        'Don': 'Mr',
        'Rev': 'Mr',
        'Dr': 'Mr',
        'Mme': 'Mrs',
        'Ms': 'Miss',
        'Major': 'Mr',
        'Lady': 'Mrs',
        'Sir': 'Mr',
        'Mlle': 'Miss',
        'Col': 'Mr',
        'Capt': 'Mr',
        'the Countess': 'Mrs',
        'Jonkheer': 'Mr',
        'Dona': 'Mrs'
    }

    Full['Title'] = Full['Name'].str.split(', ').str[1].str.split('.').str[0]
    Full['Title'] = Full['Title'].map(Map)

    #Full['Surname'] = Full['Name'].str.split(', ').str[0]

    Full['FamilySize'] = Full['SibSp'] + Full['Parch'] + 1
    Full['Age'] = Full.groupby(['Title', 'FamilySize'])['Age'].transform(lambda x: x.fillna(x.mean()))
    Full['Age'] = Full.groupby(['Title'])['Age'].transform(lambda x: x.fillna(x.mean()))

    Full['Embarked'] = Full.groupby(['Pclass'])['Embarked'].transform(lambda x: x.fillna(x.mode()[0]))

    Full['Fare'] = Full.groupby(['Pclass', 'Title'])['Fare'].transform(lambda x: x.fillna(x.mean()))

    Full['Cabin'] = Full['Cabin'].fillna('U')
    Full['Cabin'] = Full['Cabin'].apply(lambda x: 'C' if x != 'U' else x)

    AgeGroups = [0, 5, 14, 20, 35, 45, 50, 100]
    Full['AgeGroup'] = pd.cut(Full['Age'], bins = AgeGroups, labels = [0, 5, 14, 20, 35, 45, 50])

    #FareGroups = [-0.1, 7.0, 14.454, 31, 55, 1000]
    #Full['FareGroup'] = pd.cut(Full['Fare'], bins = FareGroups, labels = [0, 1, 2, 3, 4])

    Full = Full.drop(['Age', 'Fare', 'Ticket', 'Name'], axis = 1)

    Full['AgeGroup'] = Full['AgeGroup'].astype('int64')
    #Full['FareGroup'] = Full['FareGroup'].astype('int64')

    return Full.loc[Full['Train'] == True], Full.loc[Full['Train'] == False]

def CompareToFullData(Prediction, Row):
    FullDataSet = pd.read_csv("FullData.csv")
    Match = FullDataSet.loc[FullDataSet['PassengerId'] == Row['PassengerId']]
    if len(Match) == 0:
        print("No match found for " + str(Row['PassengerId']))
    else:
        Match = Match.iloc[0]
        if Match['survived'] != Prediction:
            #print("Survived values do not match for " + str(Row['PassengerId']))
            return False
        else:
            #print("Survived values match for " + str(Row['PassengerId']))
            return True

def Test(TestData, Tree):
    Correct = 0
    Incorrect = 0
    for Index in range(0, len(TestData)):
        Predict = Tree.Predict(TestData.iloc[Index])
        RowCorrect = CompareToFullData(Predict, TestData.iloc[Index])
        if RowCorrect:
            Correct += 1
        else:
            Incorrect += 1

    print("Correct: ", Correct)
    print("Incorrect: ", Incorrect)
    print("Accuracy: ", (Correct/(Correct+Incorrect))*100, "%")

TrainData, TestData = LoadData()
Start = time.time()
tree = Tree(TrainData, 5, 0.9, 2)
print("Time taken: ", time.time() - Start)

Test(TestData, tree)

