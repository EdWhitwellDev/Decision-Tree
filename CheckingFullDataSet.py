import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

# I am doing this for the Titanic problem on Kaggle
# however, as kaggle does not provide the full dataset (Survived for test is missing of course), and i don't want to submit to kaggle to check my accuracy
# I have sourced a version of the full dataset, however it doesn't cleanly match up and the passenger Ids can't be used to match them easily.
# so below is the process i have used so that i can match them up and create a usable dataset to test with.

def LoadData():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    FullTitanic3 = pd.read_excel("titanic3.xls")

    FullTitanic3 = FullTitanic3.drop(['body', 'boat', 'home.dest'], axis = 1)
    Combined = pd.concat([train, test], sort = False)
    Combined['Name'] = Combined['Name'].str.replace('"', '')

    FullTitanic3['PassengerId'] = np.arange(0, len(FullTitanic3))

    for Index in range(0, len(FullTitanic3)):
        Passenger= FullTitanic3.iloc[Index]
        Passenger['name'] = Passenger['name'].replace('"', '')
        Match = Combined.loc[Combined['Name'] == Passenger['name']]

        if len(Match) == 0:
            print("No match found for " + str(Passenger['name']))
            print(Passenger)
            print()
            # remove the quotes from the name
            PotentialMatch = Combined.loc[((Combined['Sex'] == Passenger['sex']) & (Combined['Age'] == Passenger['age']))]
            print(PotentialMatch[['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Fare', 'Pclass']])
            input()

        elif len(Match) > 1:
            #print("More than one match found for " + str(Passenger['name']))
            #print(Passenger)
            #print()
            #print(Match[['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Fare', 'Pclass']])
            Match = Match.loc[Match['Age'] == Passenger['age']]
            #rint(Match[['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Fare', 'Pclass']])
            #nput()
    
        if len(Match) == 1:
            #print(Match[['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Fare']])
            #print()
            #print(Passenger)
            FullTitanic3.at[Index, 'PassengerId'] = Match.iloc[0]['PassengerId']
            #print(FullTitanic3.iloc[Index]['PassengerId'])
            #input()
    # order by PassengerId
    FullTitanic3 = FullTitanic3.sort_values(by = ['PassengerId'])
    print(FullTitanic3.head(10))

    print()
    print(Combined.head(10))

    # save the new FullTitanic3
    FullTitanic3.to_csv("FullData.csv", index = False)
          
LoadData()