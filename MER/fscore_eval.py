
Entities=['B-MED','I-MED']

def count_Entities(y):
    total=0
    for sentence in y:
        test = 'O'
        for item in sentence:
            if (item!='O')&(test=='O'):
                total+=1
            test=item
    return total

def count_Correct(y_test,y_pred):
    countCorrect=0
    for y_testsent,y_predsent in zip(y_test,y_pred):
        correct=0
        detected=0
        for test,pred in zip(y_testsent,y_predsent):
            if (test!='O')|(pred!='O'):
                detected+=1
                if(test==pred):
                    correct+=1
            else:
                if detected!=0:
                    countCorrect+=correct/detected
                    correct=0
                    detected=0
    return countCorrect


def fscoreeval(y_test,y_pred):
    totalCount=count_Entities(y_test)
    foundCount=count_Entities(y_pred)
    countCorrect=count_Correct(y_test,y_pred)
    return countCorrect/foundCount,countCorrect/totalCount
