import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

# This was a solution i made that doesn't use machine learning, but instead uses a set of rules I found through trial and error.
# I did this to help me understand the data better, and get a reference on what was required to get above 80% accuracy

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

    Full['Title'] = Full['Name'].str.split(', ').str[1].str.split('.').str[0]
    Full['Title'] = Full['Title'].map(Map)

    Full['Surname'] = Full['Name'].str.split(', ').str[0]

    Full['Age'] = Full.groupby(['Title', 'SibSp', 'Parch'])['Age'].transform(lambda x: x.fillna(x.mean()))
    Full['Age'] = Full.groupby(['Title'])['Age'].transform(lambda x: x.fillna(x.mean()))

    Full['Embarked'] = Full['Embarked'].fillna('S')

    Full['Fare'] = Full.groupby(['Pclass', 'Title', 'Embarked'])['Fare'].transform(lambda x: x.fillna(x.mean()))

    Full['Cabin'] = Full['Cabin'].fillna('U')
    Full['Cabin'] = Full['Cabin'].apply(lambda x: 'C' if x != 'U' else x)

    AgeGroups = [0, 5, 14, 18, 35, 45, 50, 100]
    Full['AgeGroup'] = pd.cut(Full['Age'], bins = AgeGroups, labels = [0, 5, 14, 18, 35, 45, 50])

    FareGroups = [0, 7.0, 14.454, 31, 55, 1000]
    Full['FareGroup'] = pd.cut(Full['Fare'], bins = FareGroups, labels = [0, 1, 2, 3, 4])

    Full['FamilySize'] = Full['SibSp'] + Full['Parch'] + 1

    Full['ZeroFare'] = Full['Fare'].apply(lambda x: 1 if x == 0 else 0)

    FindClosestComparision(Full.loc[Full['Train'] == False], Full.loc[Full['Train'] == True])

def CheckSpecific(Row, Prediction):
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

def FindClosestComparision(Test, Train):
    Test['Correct'] = None
    for Index in range(0, len(Test)):
        Row = Test.iloc[Index]
        if Row['FamilySize'] > 4:
            Matches = Train.loc[(Train['FamilySize'] == Row['FamilySize'] ) & (Train['Pclass'] == Row['Pclass']) & (Train['Surname'] == Row['Surname'] ) & (Train['Sex'] == Row['Sex'])]
            if len(Matches) == 0:
                Matches = Train.loc[(Train['Surname'] == Row['Surname'] ) & (Train['Sex'] == Row['Sex'])]
                if len(Matches) == 0:
                    Matches = Train.loc[((Train['Pclass'] == Row['Pclass'] )& (Train['Title'] == Row['Title'] )) ]

            Predicted = Matches['Survived'].value_counts().idxmax()
            Test.at[Index, 'Survived'] = Predicted

        elif Row['Pclass'] == 3:
            Matches = Train.loc[((Train['Title'] == Row['Title']) & (Train['Pclass'] == Row['Pclass']))]
            Survived = Matches['Survived'].value_counts(normalize = True)

            if Survived[0] < 0.8:
                Matches = Train.loc[((Train['Surname'] == Row['Surname']) )]
                Survived = Matches['Survived'].value_counts(normalize = True)

                if len(Matches) == 0:
                    Matches = Train.loc[((Train['Sex'] == Row['Sex']) & (Train['Pclass'] == Row['Pclass']))]
                    Survived = Matches['Survived'].value_counts(normalize = True)
                    if Survived[0] < 0.8:
                        Matches = Train.loc[(Train['Embarked'] == Row['Embarked'])]
                        Survived = Matches['Survived'].value_counts(normalize = True)

            if len(Matches) == 0:
                Matches = Train.loc[(Train['Title'] == Row['Title'])]
                if len(Matches) == 0:
                    Matches = Train.loc[(Train['Pclass'] == Row['Pclass'])]
            Predicted = Matches['Survived'].value_counts().idxmax()
            Test.at[Index, 'Survived'] = Predicted

        elif Row['ZeroFare'] == 1:
            Matches = Train.loc[((Train['ZeroFare'] == Row['ZeroFare'])& (Train['AgeGroup'] == Row['AgeGroup'])  & (Train['Pclass'] == Row['Pclass']) )]
            if len(Matches) == 0:
                Matches = Train.loc[(Train['Pclass'] == Row['Pclass'])]
            Predicted = Matches['Survived'].value_counts().idxmax()

            Test.at[Index, 'Survived'] = Predicted

        else:
            Matches = Train.loc[((Train['Title'] == Row['Title']) & (Train['Pclass'] == Row['Pclass']) )]
            if len(Matches) == 0:
                Matches = Train.loc[(Train['Pclass'] == Row['Pclass'])]
            Predicted = Matches['Survived'].value_counts().idxmax()

            Test.at[Index, 'Survived'] = Predicted
        Test.at[Index, 'Correct'] = CheckSpecific(Row, Test.at[Index, 'Survived'])

    print(Test['Correct'].value_counts(normalize = True))

LoadData()