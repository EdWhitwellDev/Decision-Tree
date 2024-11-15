import pandas as pd
import numpy as np
import time

from statistics import mode

pd.options.mode.chained_assignment = None  # default='warn'

# New method finally achieves 80% accuracy

class Layer:
    def __init__(self, UseableAttributes, DataPassedDown, ExclusiveAttributes, Prediction = 0, BenchMark = 0.1, Depth = 0, MinSize = 3, MaxDepth = 3, Threshold = 0.45, CallIt = 1):
        self.Prediction = Prediction
        self.End = True
        self.BenchMark = BenchMark
        if Depth == MaxDepth:
            self.End = False
            return
        
        DataPassedDown = DataPassedDown.reset_index(drop = True)
        Data = DataPassedDown.copy()

        ViablePrediction = np.zeros(len(Data))

        self.UseSolid, NotSoGood = [], []
 
        for Attribute in UseableAttributes:
            TrueFor = Data.loc[Data[Attribute] == True]
            if len(TrueFor) < MinSize:
                continue
            Survived = sum(TrueFor['Survived'])
            Ratio = Survived/len(TrueFor)
            Adjusted = abs(Ratio - 0.5)
            if Adjusted > Threshold:
                self.UseSolid.append([Attribute, Adjusted, round(Ratio)])
                ViablePrediction[TrueFor.index] = 1
            elif Adjusted > BenchMark:
                NotSoGood.append([Attribute, Adjusted, round(Ratio), TrueFor.index])

        NotSoGood.sort(key = lambda x: x[1], reverse = True)
        NotSolid, self.NextLayers = 0, []

        while sum(ViablePrediction) < round(len(Data) * CallIt) and NotSolid < len(NotSoGood):
            ViablePrediction[NotSoGood[NotSolid][3]] = 1
            AttributesToUse = self.StripExclusiveAttributes(UseableAttributes.copy(), NotSoGood[NotSolid][0], ExclusiveAttributes)
            self.NextLayers.append(Layer(AttributesToUse, DataPassedDown.iloc[NotSoGood[NotSolid][3]], ExclusiveAttributes, NotSoGood[NotSolid][2], NotSoGood[NotSolid][1], Depth+1))
            NotSolid += 1
        self.UseNotSolid = NotSoGood[:NotSolid]

    def StripExclusiveAttributes(self, AttributeList, Attribute, ExclusiveAttributes = {}):
        Core = Attribute.split('_')[0]
        try:
            Exclusive = ExclusiveAttributes[Core]
            AttributeList = AttributeList.drop(Exclusive)
        except:AttributeList = AttributeList.drop(Attribute)
        return AttributeList

    def Predict(self, Passenger):
        Prediction = [0, 0]
        Prediction[self.Prediction] = Prediction[self.Prediction] + self.BenchMark
        if self.End:
            for Index in range(len(self.UseSolid)):
                if Passenger[self.UseSolid[Index][0]] == True: Prediction[self.UseSolid[Index][2]] += self.UseSolid[Index][1] * 2
            for Index in range(len(self.UseNotSolid)):
                if Passenger[self.UseNotSolid[Index][0]] == True: Prediction = np.add(Prediction, self.NextLayers[Index].Predict(Passenger))
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

    Full['FamilySize'] = Full['SibSp'] + Full['Parch'] + 1

    Full['Age'] = Full.groupby(['Title', 'FamilySize'])['Age'].transform(lambda x: x.fillna(x.mean()))
    Full['Age'] = Full.groupby(['Title'])['Age'].transform(lambda x: x.fillna(x.mean()))

    Full['Embarked'] = Full.groupby(['Pclass'])['Embarked'].transform(lambda x: x.fillna(x.mode()[0]))

    Full['Fare'] = Full.groupby(['Pclass', 'Title', 'Embarked'])['Fare'].transform(lambda x: x.fillna(x.mean()))

    Full['Cabin'] = Full['Cabin'].notnull()

    AgeGroups = [-0.1, 2, 12, 18, 30, 60, np.inf]
    labels = ['baby', 'child', 'teenager', 'youngadult', 'adult', 'elderly']
    Full['AgeGroup'] = pd.cut(Full['Age'], bins = AgeGroups, labels = labels)

    Full['Mother'] = (Full['Sex'] == 'female') & (Full['Title'] == 'Mrs') & (Full['Parch'] > 0) & (Full['AgeGroup'].isin(['youngadult', 'adult']))
    Full['Daughter'] = (Full['Sex'] == 'female') & (Full['Title'] == 'Miss') & (Full['AgeGroup'].isin(['baby', 'child', 'teenager'])) & (Full['Parch'] > 0)

    Full['FareGroup'] = pd.qcut(Full['Fare'], 5)

    Full = Full.drop(['Age', 'Fare', 'Ticket', 'Name'], axis = 1)

    UsableColumns = Full.columns.copy().drop(['PassengerId', 'Survived', 'Train', 'Cabin', 'Mother'])
    Exclusive = {}
    for Attribute in UsableColumns:
        Exclusive[Attribute] = []
        Full = pd.concat([Full, pd.get_dummies(Full[Attribute], prefix = Attribute)], axis = 1)
        Full = Full.drop([Attribute], axis = 1)
        Exclusive[Attribute] = Full.columns[Full.columns.str.startswith(Attribute)]

    UsableColumns = Full.columns.copy().drop(['PassengerId', 'Survived', 'Train'])

    for Attribute in UsableColumns:
        if Attribute != 'PassengerId':
            Full[Attribute] = Full[Attribute].astype('bool')

    return Full.loc[Full['Train'] == True], Full.loc[Full['Train'] == False], Exclusive

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

TrainData, TestData, Exclusive = LoadData()

UsableColumns = TrainData.columns.copy().drop(['PassengerId', 'Survived', 'Train'])
Start = time.time()
FirstLayer = Layer(UsableColumns, TrainData, Exclusive)
print("Completed In: ", time.time() - Start)
Test(TestData, FirstLayer)