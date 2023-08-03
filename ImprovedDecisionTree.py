import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

# this is what i built after i found out what Gini was, its a slight improvement in accuracy over the first attempt and a bit faster

# looking back on it now before i upload it to github, this isn't my best coding its really messy if you do plan on looking through it.

# i'm not sure if decision trees are supposed to spit the data set by every category every time but this does and i dont think thats great
# i actively tried to do as little research on decision trees so that i could try to figure them out on my own as much as possible, as they are supposed to be straight forward

# i get around 77.3% accuracy with this version

class Forrest:
    def __init__(self, NumberOfTrees, TreeDepth, MinSize):
        self.NumberOfTrees = NumberOfTrees
        self.TreeDepth = TreeDepth
        self.MinSize = MinSize

        self.Trees = []

    def Train(self, Data):
        StartAttribute = None
        for Index in range(0, self.NumberOfTrees):
            print("Training tree " + str(Index+1), end = '\r')

            SubData = Data.sample(frac = 0.75, replace = False)
            NewTree = Tree(self.TreeDepth, self.MinSize, SubData, StartAttribute)
            StartAttribute = NewTree.Root.SplitAttribute
            self.Trees.append(NewTree)
        print()

    def Predict(self, Data):
        Predictions = 0
        for Index in range(0, self.NumberOfTrees):

            Predictions += self.Trees[Index].Predict(Data, False)
        Predictions = Predictions / self.NumberOfTrees
        #print(Predictions)
        Predictions = np.round(Predictions)
        #print(Predictions)

        return Predictions
        

class Tree:
    def __init__(self, MaxDepth, MinSize, Data, StartAttribute = None):
        self.MaxDepth = MaxDepth
        self.MinSize = MinSize

        self.Attributes = Data.columns.tolist()

        self.Root = Node(self, StartAttribute, 0, Data)

    def Predict(self, Data, Round = True):
        Predictions = self.Root.Predict(Data)
        # round the predictions
        if Round:
            Predictions = np.round(Predictions)
        return Predictions

        
class Node:
    def __init__(self, Tree, Attribute, Depth, Data):
        self.Tree = Tree
        self.Attribute = Attribute
        self.Depth = Depth
        self.Branches = []

        self.Burned = False

        if len(Data) < Tree.MinSize:
            if self.Attribute == 'Surname':
                if len(Data) == 0:
                    self.Burned = True
                    return None
                else:
                    self.ProbSurvived = Data['Survived'].sum() / Data.shape[0]
                    return None
            #print("Too small, burning")
            self.Burned = True

            return None

        ProbSurvived = Data['Survived'].sum() / Data.shape[0]
        ProbDead = 1 - ProbSurvived
        self.Gini = 1 - (ProbSurvived * ProbSurvived) - (ProbDead * ProbDead)
        if self.Gini == 0:
            self.ProbSurvived = ProbSurvived
            return None
        self.ProbSurvived = ProbSurvived
        self.Matches = Data['Survived'].copy()

        if Depth == Tree.MaxDepth:
            pass

        else:
            BestGini, SplitAttribute = self.FindBestSplit(Data)
            if BestGini < self.Gini:
                self.Burned = False
                self.SplitAttribute = SplitAttribute
                self.Split(Data)
            else:
                self.Burned = True

    def Predict(self, Data):
        if self.Burned:
            return None

        if len(self.Branches) == 0:
            return self.ProbSurvived
        
        ItemCat = Data[self.SplitAttribute]
        if ItemCat not in self.Cats:
            return self.ProbSurvived
        Branch = self.Branches[self.Cats.index(ItemCat)]
        predict = Branch.Predict(Data)
        if predict == None:
            return self.ProbSurvived
        return predict
    
    def Split(self, Data):
        Groups = Data.groupby(self.SplitAttribute)
        Cats = list(Groups.groups.keys())
        self.Cats = Cats
        for Index in range(0, len(Cats)):
            Branch = Data.loc[Data[self.SplitAttribute] == Cats[Index]]
            NewNode = Node(self.Tree, self.SplitAttribute, self.Depth + 1, Branch)
            self.Branches.append(NewNode)

    def FindBestSplit(self, Data):
        BestSplit = None
        BestGini = 1.0
        BestPerfectCatageory = 0.25
        PerfectishCatageory = None
        for Attribute in self.Tree.Attributes:
            if Attribute == 'Survived' or Attribute == 'isTrain' or Attribute == self.Attribute or Attribute == 'PassengerId':
                continue

            Groups = Data.groupby(Attribute, observed = True)
            GroupsSurvived = Groups['Survived'].value_counts(normalize = True).unstack()
            GroupsCats = GroupsSurvived.index.tolist()
            GroupProportions = Groups.size() / Data.shape[0]
            Gini = 0.0
            PerfectCatageory = 0.0
            for Index in range(0, len(GroupsSurvived)):
                Cat = GroupsCats[Index]
                CurrentGini = GroupProportions.iloc[Index] * (1 - (GroupsSurvived[0][Cat] * GroupsSurvived[0][Cat]) - (GroupsSurvived[1][Cat] * GroupsSurvived[1][Cat]))
                if CurrentGini < 0.01:
                    PerfectCatageory += GroupProportions.iloc[Index]
                Gini += CurrentGini

            if Gini < BestGini:
                BestSplit = Attribute
                BestGini = Gini
            if PerfectCatageory > BestPerfectCatageory:
                BestPerfectCatageory = PerfectCatageory

                PerfectishCatageory = Attribute

        if PerfectishCatageory != None:
            BestSplit = PerfectishCatageory
            BestGini = 0.0

        return BestGini, BestSplit

def LoadData():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    test['Survived'] = None
    train['isTrain'] = True
    test['isTrain'] = False

    Combined = pd.concat([train, test], sort = False)

    Map = {
        'Mr': 'Mr',
        'Mrs': 'Mrs',
        'Miss': 'Miss',
        'Master': 'Master',
        'Don': 'Mr',
        'Rev': 'Rev',
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

    Combined['Title'] = Combined['Name'].str.split(', ').str[1].str.split('.').str[0]
    Combined['Title'] = Combined['Title'].map(Map)

    Combined['Surname'] = Combined['Name'].str.split(', ').str[0]

    Combined['Age'] = Combined.groupby(['Title', 'SibSp', 'Parch'])['Age'].transform(lambda x: x.fillna(x.mean()))
    Combined['Age'] = Combined.groupby(['Title'])['Age'].transform(lambda x: x.fillna(x.mean()))

    Combined['Embarked'] = Combined['Embarked'].fillna('S')

    Combined['Fare'] = Combined.groupby(['Pclass', 'Title', 'Embarked'])['Fare'].transform(lambda x: x.fillna(x.mean()))

    Combined['Cabin'] = Combined['Cabin'].fillna('U')
    Combined['Cabin'] = Combined['Cabin'].apply(lambda x: 'C' if x != 'U' else x)

    AgeGroups = [0, 5, 14, 18, 35, 45, 50, 100]
    Combined['AgeGroup'] = pd.cut(Combined['Age'], bins = AgeGroups, labels = [0, 5, 14, 18, 35, 45, 50])

    FareGroups = [0, 7.0, 14.454, 31, 55, 1000]
    Combined['FareGroup'] = pd.cut(Combined['Fare'], bins = FareGroups, labels = [0, 1, 2, 3, 4])

    Combined['FamilySize'] = Combined['SibSp'] + Combined['Parch'] + 1

    Combined['ZeroFare'] = Combined['Fare'].apply(lambda x: 1 if x == 0 else 0)


    Combined = Combined.drop(['Age', 'Fare', 'Ticket', 'Name'], axis = 1)

    return Combined.loc[Combined['isTrain'] == True], Combined.loc[Combined['isTrain'] == False]

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
    print()
    print("Correct: " + str(Correct))
    print("Incorrect: " + str(Incorrect))
    print("Accuracy: " + str((Correct*100) / (Correct + Incorrect)))

Train, TestData = LoadData()
TestForrest = Forrest(25, 10, 4)
TestForrest.Train(Train)
Test(TestData, TestForrest)
