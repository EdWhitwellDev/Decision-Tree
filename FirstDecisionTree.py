import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


# this was my first attempt at Decision Trees that i made without doing any research
# its very slow as the way it works is by finding a critical point which it gets by trying values between the average value of the attribute for the survived and died passengers until the ratio between Survived and Died gets worse
# it then splits the data into two groups and repeats the process
# it is a little different for catagorical data as it finds the catagory that has the biggest difference between the number of survived and died passengers

# there are a few problems with this approach, the main one is that it is not greedy with the attributes it chooses but instead uses a predefined order
# i did it this way as it would take too long to try each different attribute with the method i was using
# another problem is that it only does binary splits, which isn't necessaraly a problem but i think it does loose some potential

# it doesn't get a great accuracy, i remember strapping a neural network to it but i think that just caused overfitting so i removed it
# it should get around 76% but depends a lot on the order 

class Forest:
    def __init__(self, Divisions):
        self.Divisions = 70
        self.Trees = []
        self.NoTrees = 25
        return
    
    def CheckIfAlreadyATree(self, AttOrder, NewOrder):
        for Order in AttOrder:
            if np.array_equal(Order, NewOrder):
                return True
        return False
    
    def Grow(self, Dead, Survived):
        NoTrees = self.NoTrees
        self.Atributes = Dead.columns.values
        AttOrder = []
        for Index in range(0, NoTrees):
            Flag = True
            while Flag:
                Shuffle = np.random.permutation(len(self.Atributes))
                self.Atributes = self.Atributes[Shuffle]
                Flag = self.CheckIfAlreadyATree(AttOrder, self.Atributes)
            AttOrder.append(self.Atributes.copy())
        AttributeOrders = np.array(AttOrder)

        for i in range(0, len(AttributeOrders)):
            print("Growing Tree: ", i+1, "/",len(AttributeOrders), end='\r')
            self.Trees.append(Tree(self.Divisions, AttributeOrders[i]))
            self.Trees[i].Grow(Dead, Survived)

    def Predict(self, Data):
        Predictions = np.empty(len(Data))
        for i in range(0, len(self.Trees)):
            Predictions[i] = self.Trees[i].Predict(Data)

        if np.sum(Predictions) > len(self.Trees)/2:
            return 1
        else:
            return 0
        
    def PredictBulkRaw(self, Data, Round = True):
        Predictions = np.zeros((len(Data), len(self.Trees)))
        for i in range(0, len(self.Trees)):
            Predictions[:,i] = self.Trees[i].PredictBulk(Data)
        return Predictions
        
    def PredictBulk(self, Data, Round = True):
        self.Predictions = np.zeros(len(Data))
        for i in range(0, len(self.Trees)):
            self.Predictions += self.Trees[i].PredictBulk(Data)
        self.Predictions[self.Predictions <= (len(self.Trees)/2 )] = 0
        self.Predictions[self.Predictions > (len(self.Trees)/2) ] = 1

        return self.Predictions

class Tree:
    def __init__(self, Divisions, Atributes):
        self.Divisions = Divisions
        self.Atributes = Atributes
        self.Root = None
        return
    
    def Grow(self, Dead, Survived):
        self.Ratio = len(Dead) / len(Survived)
        self.Root = Node(self, 0, Dead, Survived)
        return
    
    def Predict(self, Data):
        return self.Root.Predict(Data)
    
    def PredictBulk(self, Data, Round = True):
        Data.reset_index(drop=True, inplace=True)
        self.Predictions = np.zeros(len(Data))
        self.Root.PredictBulk(Data)
        return self.Predictions

class Node:
    def __init__(self, Tree, Level, Dead, Survived):
        self.Tree = Tree
        self.Key = Tree.Atributes[Level]
        self.Level = Level


        if len(Dead) == 0 or len(Survived) == 0:
            self.Left = None
            self.Right = None
            self.CriticalPoint = None
            self.Probability = 0.5
            self.NextLevel = None
            self.Level = None
            if len(Dead) == 0:
                self.Probability = 1
            elif len(Survived) == 0:
                self.Probability = 0
            return

        if Level == None:
            self.Level = None
            self.Left = None
            self.Right = None
            Survived = len(Survived)
            Dead = len(Dead)
            Percentage = Survived / (Survived + Dead)
            self.Probability = Percentage
            return
        
        if Level < len(Tree.Atributes)-1:
            self.NextLevel = Level + 1
        else:
            self.NextLevel = None

        if Dead[self.Key].dtype == np.float64 or Dead[self.Key].dtype == np.int64:
            self.CriticalPoint = self.FindCriticalPoint(Dead, Survived)
        else:
            self.CriticalPoint = self.MostSignificantCatagory(Dead, Survived)

        DeadLeft = Dead[Dead[self.Key] < self.CriticalPoint]
        SurvivedLeft = Survived[Survived[self.Key] < self.CriticalPoint]
        DeadRight = Dead[Dead[self.Key] >= self.CriticalPoint]
        SurvivedRight = Survived[Survived[self.Key] >= self.CriticalPoint]
        self.Left = Node(Tree, self.NextLevel, DeadLeft, SurvivedLeft)
        self.Right = Node(Tree, self.NextLevel, DeadRight, SurvivedRight)

        return 
    
    def Predict(self, Data):
        if self.NextLevel == None:
            return self.Probability
        if Data[self.Key] < self.CriticalPoint:
            return self.Left.Predict(Data)
        else:
            return self.Right.Predict(Data)
        
    def PredictBulk(self, Data):
        if self.Level == None:
            self.Tree.Predictions[Data.index] = self.Probability
            return 

        LeftData = Data[Data[self.Key] < self.CriticalPoint]
        RightData = Data[Data[self.Key] >= self.CriticalPoint]

        self.Left.PredictBulk(LeftData)
        self.Right.PredictBulk(RightData)

    def FindCriticalPoint(self, Dead, Survived):
        Smaller = min(Survived[self.Key].mean(), Dead[self.Key].mean())
        Bigger = max(Survived[self.Key].mean(), Dead[self.Key].mean())
        Increment = (Bigger - Smaller) / self.Tree.Divisions
        NumberOfDead = 0
        NumberOfSurvived = 0
        SurvivedBigger = False
        if Survived[self.Key].mean() > Dead[self.Key].mean():
            SurvivedBigger = True
        for i in range(0, self.Tree.Divisions):
            Region = Smaller + i * Increment
            NumberOfSurvivedNew = np.sum(Survived[self.Key] < Region) 
            NumberOfDeadNew = np.sum(Dead[self.Key] < Region) 
            ChangeInSurvived = (NumberOfSurvivedNew - NumberOfSurvived) / len(Survived)
            ChangeInDead = (NumberOfDeadNew - NumberOfDead) / len(Dead)
            NumberOfSurvived = NumberOfSurvivedNew
            NumberOfDead = NumberOfDeadNew
            if SurvivedBigger:
                if ChangeInSurvived > ChangeInDead:
                    return Region
                
            else:
                if ChangeInSurvived < ChangeInDead:
                    return Region
                
        return Smaller
    
    def MostSignificantCatagory(self, Dead, Survived):
        DeadPerCatagory = Dead[self.Key].value_counts()
        SurvivedPerCatagory = Survived[self.Key].value_counts()
        DifferencePerCatagory = SurvivedPerCatagory - DeadPerCatagory
        AbsDifferencePerCatagory = DifferencePerCatagory.abs()
        return AbsDifferencePerCatagory.idxmax()

def LoadDataAndPreprocess():
    print("Loading and Preprocessing Data")

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    train['Title'] = train['Name'].str.split(', ').str[1].str.split('.').str[0]
    test['Title'] = test['Name'].str.split(', ').str[1].str.split('.').str[0]

    train['Surname'] = train['Name'].str.split(', ').str[1].str.split('. ').str[1].str.split(' ').str[0]
    test['Surname'] = test['Name'].str.split(', ').str[1].str.split('. ').str[1].str.split(' ').str[0]

    train.loc[train['Surname'].value_counts()[train['Surname']].values == 1, 'Surname'] = 'Unique'
    test.loc[test['Surname'].value_counts()[test['Surname']].values == 1, 'Surname'] = 'Unique'

    train['Cabin'].fillna(value='N23 N23', inplace = True)
    train['Platfrom'] = train['Cabin'].str.split(' ').str[0].str[0]
    test['Cabin'].fillna(value='N23 N23', inplace = True)
    test['Platfrom'] = test['Cabin'].str.split(' ').str[0].str[0]

    train.drop(columns = ['Name','Ticket','Cabin'], inplace = True)
    test.drop(columns = ['Name','Ticket','Cabin'], inplace = True)

    train['Embarked'] = train.groupby(['Survived', 'Pclass'])['Embarked'].transform(lambda x: x.fillna(x.mode()[0]))

    test['Embarked'].fillna(value='S', inplace = True)

    train['Age'] = train.groupby(['SibSp', 'Title'])['Age'].transform(lambda x: x.fillna(x.mean()))

    train['Mother'] = (train['Sex'] == 'female') & (train['Title'] != 'Miss') & (train['Parch'] > 0)
    test['Mother'] = (test['Sex'] == 'female') & (test['Title'] != 'Miss')  & (test['Parch'] > 0)
    train['Daughter'] = (train['Sex'] == 'female') & (train['Title'] == 'Miss') & (train['Age'] < 20) & (train['Parch'] > 0)
    test['Daughter'] = (test['Sex'] == 'female') & (test['Title'] == 'Miss') & (test['Age'] < 20) & (test['Parch'] > 0)

    test['Age'] = train.groupby(['Title', 'SibSp'])['Age'].transform(lambda x: x.fillna(x.mean()))
    test['Age'] = train.groupby(['Title'])['Age'].transform(lambda x: x.fillna(x.mean()))
    

    test['Fare'] = train.groupby(['Pclass', 'SibSp', 'Title'])['Fare'].transform(lambda x: x.fillna(x.mean()))


    return train, test

def CompareToFullData(Data):
    FullDataSet = pd.read_csv("FullData.csv")
    Correct = 0
    Incorrect = 0
    
    for Index in range(0, len(Data)):
        Row = Data.iloc[Index]
        Match = FullDataSet.loc[FullDataSet['PassengerId'] == Row['PassengerId']]
        if len(Match) == 0:
            print("No match found for " + str(Row['PassengerId']))
        else:
            Match = Match.iloc[0]
            if Match['survived'] != Row['Survived']:
                Incorrect += 1
            else:
                Correct += 1

    print("Correct: " + str(Correct))
    print("Incorrect: " + str(Incorrect))
    print("Accuracy: " , (Correct / (Correct + Incorrect))*100)

def SplitSurvivedAndDied(Data, test):
    print("Splitting Survived and Died")

    Survived = Data.loc[Data['Survived'] == 1]
    Died = Data.loc[Data['Survived'] == 0]

    Survived.drop(columns = ['Survived', 'PassengerId'], inplace = True)
    Died.drop(columns = ['Survived', 'PassengerId'], inplace = True)

    Died['SibSp'] = Died['SibSp'].astype(float)
    Survived['SibSp'] = Survived['SibSp'].astype(float)

    return Survived, Died, test

def CreateSubmition(Forest, test):
    Predictions = Forest.PredictBulk(test).astype(int)
    Submition = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': Predictions})
    CompareToFullData(Submition)
    Submition.to_csv('Submition.csv', index=False)
    return

train, test = LoadDataAndPreprocess()
Survived, Died, test = SplitSurvivedAndDied(train, test)
TestForest = Forest(100)
TestForest.Grow(Died, Survived)

Dead = TestForest.PredictBulk(Died)
Alive = TestForest.PredictBulk(Survived)

Total = len(Dead) + len(Alive)
Correct = len(Dead) - np.sum(Dead) + np.sum(Alive)
print()
print("Total Correct on training: ", Correct, "/", Total, " = ", Correct/Total)

print("Testing")
CreateSubmition(TestForest, test)

