I have attempted to "solve" the titanic problem using Decision trees
I didn't do much research into how they worked and tried to learn by developing them
I started by trying a few simple things, like grouping by a hierarchy of attributes,
Then i hard coded splits/predictions instead of using a machine learning technique

The first Tree i made, doesn't use gini or entropy, not because i used something better but because i didn't know what it was, it only did binary splits, and determined the critical value by trying different continuous values until the ratio of suvived and died got worse. a forest of 25 got 75-76% accuracy.

the next used gini and only categorical attributes, it got to around 77% again with 25 trees

the last one i am still working on. instead of only passing split data down the branches as i did with the other one, if the split does not improve the accuracy for a catagory, the complete data is is passed down its branch, however the branch then decides the next attribute based on how well it predicts the previously poorly predicted catagory. I am still developing this one, however it gets 79.2% accuracy with only 1 tree, i hope to get it to 80% with only one tree, but failing that i will use a forest with some kind of boosting.   
 