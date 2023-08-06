import pandas as pd
import numpy as np
import time

from statistics import mode

pd.options.mode.chained_assignment = None  # default='warn'

class Layer:
    def __init__(self, UseableAttributes, DataPassedDown, Prediction = 0, BenchMark = 0.0, Depth = 0, MinSize = 2, MaxDepth = 3, Threshold = 0.48, CallIt = 1):
        self.Prediction = Prediction
        self.End = True
        self.BenchMark = BenchMark
        if Depth == MaxDepth:
            self.End = False
            return

        Data = DataPassedDown.copy()
        Data['ViablePrediction'] = 0

        self.Solid, self.NotSoGood = [], []
 
        for Attribute in UseableAttributes:
            TrueFor = Data.loc[Data[Attribute] == True]
            if len(TrueFor) < MinSize:
                continue
            Survived = TrueFor.loc[TrueFor['Survived'] == 1]
            Ratio = len(Survived)/len(TrueFor)
            Adjusted = abs(Ratio - 0.5)
            if Adjusted > Threshold:
                self.Solid.append([Attribute, Adjusted, round(Ratio)])
            elif Adjusted > BenchMark:
                self.NotSoGood.append([Attribute, Adjusted, round(Ratio)])

        self.Solid.sort(key = lambda x: x[1], reverse = True)
        self.NotSoGood.sort(key = lambda x: x[1], reverse = True)

        self.SolidPrediction = []

        SolidIndex, NotSolid = 0, 0
        while sum(Data['ViablePrediction']) < round(len(Data) * CallIt) and SolidIndex < len(self.Solid):
            Data.loc[Data[self.Solid[SolidIndex][0]] == True, 'ViablePrediction'] = 1
            self.SolidPrediction.append(self.Solid[SolidIndex][2])
            SolidIndex += 1

        while sum(Data['ViablePrediction']) < round(len(Data) * CallIt) and NotSolid < len(self.NotSoGood):
            Data.loc[Data[self.NotSoGood[NotSolid][0]] == True, 'ViablePrediction'] = 1
            NotSolid += 1

        self.UseSolid = self.Solid[:SolidIndex]

        self.UseNotSolid = self.NotSoGood[:NotSolid]

        self.NextLayers = []


        for Index in range(len(self.UseNotSolid)):
            AttributesToUse = UseableAttributes.copy().drop([self.UseNotSolid[Index][0]])
            DataToUse = DataPassedDown.loc[Data[self.UseNotSolid[Index][0]] == True]
            NewLayer = Layer(AttributesToUse, DataToUse, self.UseNotSolid[Index][2], self.UseNotSolid[Index][1], Depth+1)
            self.NextLayers.append(NewLayer)

    def Predict(self, Passenger):
        Prediction = [0, 0]
        Prediction[self.Prediction] = Prediction[self.Prediction] + self.BenchMark
        if self.End:
            for Index in range(len(self.UseSolid)):
                if Passenger[self.UseSolid[Index][0]] == True:
                    Prediction[self.UseSolid[Index][2]] += self.UseSolid[Index][1] * 2.5

            for Index in range(len(self.UseNotSolid)):
                if Passenger[self.UseNotSolid[Index][0]] == True:
                    Prediction = np.add(Prediction, self.NextLayers[Index].Predict(Passenger))
                    #print(Prediction)
            
        return Prediction

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
    Full['Mother'] = (Full['Sex'] == 'female') & (Full['Title'] == 'Mrs') & (Full['Parch'] > 0)
    Full['Daughter'] = (Full['Sex'] == 'female') & (Full['Title'] == 'Miss') & (Full['Age'] < 16) & (Full['Parch'] > 0)

    #Full['Cabin'] = Full['Cabin'].fillna('U')
    #Full['Cabin'] = Full['Cabin'].apply(lambda x: 'C' if x != 'U' else x)
    Full = Full.drop(['Cabin'], axis =1 )

    AgeGroups = [0, 5, 14, 20, 35, 45, 50, 100]
    Full['AgeGroup'] = pd.cut(Full['Age'], bins = AgeGroups, labels = [0, 5, 14, 20, 35, 45, 50])

    FareGroups = [-0.1, 7.0, 14.454, 31, 55, 1000]
    Full['FareGroup'] = pd.cut(Full['Fare'], bins = FareGroups, labels = [0.1, 1, 2, 3, 4])

    Full = Full.drop(['Age', 'Fare', 'Ticket', 'Name'], axis = 1)


    Full['AgeGroup'] = Full['AgeGroup'].astype('int64')
    Full['FareGroup'] = Full['FareGroup'].astype('int64')

    #Full['LargeFamily'] = Full['FamilySize'] > 4

    UsableColumns = Full.columns.copy().drop(['PassengerId', 'Survived', 'Train'])
    for Attribute in UsableColumns:
        Full = pd.concat([Full, pd.get_dummies(Full[Attribute], prefix = Attribute)], axis = 1)
        Full = Full.drop([Attribute], axis = 1)

    UsableColumns = Full.columns.copy().drop(['PassengerId', 'Survived', 'Train'])

    for Attribute in UsableColumns:
        if Attribute != 'PassengerId':
            Full[Attribute] = Full[Attribute].astype('bool')

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
        Predict = np.array(Tree.Predict(TestData.iloc[Index])).argmax()
        RowCorrect = CompareToFullData(Predict, TestData.iloc[Index])
        if RowCorrect:
            Correct += 1
        else:
            Incorrect += 1

    print("Correct: ", Correct)
    print("Incorrect: ", Incorrect)
    print("Accuracy: ", (Correct/(Correct+Incorrect))*100, "%")

TrainData, TestData = LoadData()

UsableColumns = TrainData.columns.copy().drop(['PassengerId', 'Survived', 'Train'])

FirstLayer = Layer(UsableColumns, TrainData)

Test(TestData, FirstLayer)