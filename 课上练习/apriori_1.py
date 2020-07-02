from akaprriori import Apriori

dataset = [("ariocot","apple","cherry","plum","banana"),("strawbeery","plum","cherry"),("persimmon","peach","banana","apple"),
           ("kiwifuit","apple","pear"),("cherry","pear","banana"),("watermelon","apple")]

rules = apriori(dataset,support = 0.05,confidence = 0.3,lift = 2)
rules_sorted = sortd(rules,key = lambda x:[x[4],x[3],x[2]],reverse = True)