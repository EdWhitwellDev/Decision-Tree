import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  

# this was the niaive approach to the problem
# the idea is i rank the attributes by how much they improve the accuracy of the prediction
# then simply go down the list trying to match the Test passenger to passengers in the training set
# it doesn't get a good accuracy however its very basic

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

def LoadData():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    train['Title'] = train['Name'].str.split(', ').str[1].str.split('.').str[0]
    test['Title'] = test['Name'].str.split(', ').str[1].str.split('.').str[0]

    train['Title'] = train['Title'].map({'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master', 'Don': 'Mr', 'Rev': 'Rev', 'Dr': 'Mr', 'Mme': 'Mrs', 'Ms': 'Miss', 'Major': 'Mr', 'Lady': 'Mrs', 'Sir': 'Mr', 'Mlle': 'Miss', 'Col': 'Mr', 'Capt': 'Mr', 'Countess': 'Mrs', 'Jonkheer': 'Mr', 'Dona': 'Mrs'})
    test['Title'] = test['Title'].map({'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master', 'Don': 'Mr', 'Rev': 'Rev', 'Dr': 'Mr', 'Mme': 'Mrs', 'Ms': 'Miss', 'Major': 'Mr', 'Lady': 'Mrs', 'Sir': 'Mr', 'Mlle': 'Miss', 'Col': 'Mr', 'Capt': 'Mr', 'Countess': 'Mrs', 'Jonkheer': 'Mr', 'Dona': 'Mrs'})

    ageGroups = [0, 1, 3, 5, 8, 12, 15, 18, 22, 30, 40, 55, 65, 80]

    train['AgeEstimated'] = train['Age'].map(lambda x: 1 if ((x % 1 == 0.5) & (x > 1.0)) else 0)
    test['AgeEstimated'] = test['Age'].map(lambda x: 1 if x % 1 == 0.5 else 0)

    train['Age'] = train.groupby(['Title', 'Parch'])['Age'].transform(lambda x: x.fillna(x.mean()))
    train['Age'] = train.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))

    test['Age'] = test.groupby(['Title', 'Parch'])['Age'].transform(lambda x: x.fillna(x.mean()))
    test['Age'] = test.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))

    train['AgeGroup'] = pd.cut(train['Age'], ageGroups)
    test['AgeGroup'] = pd.cut(test['Age'], ageGroups)

    FareGroups = [0, 7.91, 14.454, 31, 512.3292]
    train['FareGroup'] = pd.cut(train['Fare'], FareGroups)
    test['FareGroup'] = pd.cut(test['Fare'], FareGroups)

    return train, test

def Prediction(Train, Test):
    RefineOrder = ['Title', 'Pclass', 'AgeGroup', 'SibSp', 'Parch', 'FareGroup', 'Embarked']
    NoMatches = 0
    Submissions = pd.DataFrame({'PassengerId': Test['PassengerId'], 'Survived': 0})
    for Index in Test.index:
        Row = Test.loc[Index]
        Refined = Train.loc[(Train['Title'] == Row['Title'])]
        Attri = 1
        SurviDie = Refined['Survived'].value_counts(sort = False)

        if len(SurviDie) == 1:
            ProbabilityDead = 0
        else:
            ProbabilityDead = SurviDie[0] / (SurviDie[0] + SurviDie[1])

        while len(Refined) >= 5 and (ProbabilityDead < 0.9 and ProbabilityDead > 0.1) and Attri < len(RefineOrder):

            RefinedTemp = Refined.loc[(Refined[RefineOrder[Attri]] == Row[RefineOrder[Attri]])]
            SurviDie = RefinedTemp['Survived'].value_counts()

            if len(SurviDie) == 1:
                ProbabilityDead = 0
                
            elif len(SurviDie) == 2:
                ProbabilityDead = SurviDie[0] / (SurviDie[0] + SurviDie[1])
                Refined = Refined.loc[(Refined[RefineOrder[Attri]] == Row[RefineOrder[Attri]])]
            Attri += 1

        if len(Refined) == 0:
            NoMatches += 1
        else:
            Submissions['Survived'].loc[Index] = Refined['Survived'].value_counts().idxmax()

    Submissions.to_csv("Submissions.csv", index=False)
    CompareToFullData(Submissions)
    print("No matches found:", NoMatches, "/", len(Test.index))

Train, Test = LoadData()
Prediction(Train, Test)
