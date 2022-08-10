import json 
import sys


def checkResult(labels, prediction):
		
    correct_n = 0 
    union_n = 0 
    #intersection_n = 0 
    correct_pos_n = 0
    label_pos_n = 0
    pred_pos_n = 0 

    for i in range(len(prediction)):
        if prediction[i] == labels[i]:
            correct_n += 1

        if prediction[i] == labels[i] and prediction[i] == 1:
            correct_pos_n += 1

        if prediction[i] + labels[i] >= 1:
            union_n += 1

        if prediction[i] == 1:
            pred_pos_n += 1
        if labels[i] == 1:
            label_pos_n += 1
    
    print(correct_n, correct_pos_n, union_n, label_pos_n, pred_pos_n, len(prediction))

    print("Accuracy ", float(correct_n) / len(prediction))
    print("Precision ", float(correct_pos_n) / (pred_pos_n + 0.001) )
    print("Recall ", float(correct_pos_n) / (label_pos_n + 0.001) )

    p = float(correct_pos_n) / (pred_pos_n + 0.001)
    r = float(correct_pos_n) / (label_pos_n + 0.001)

    print("F1", 2*p*r/(p+r+0.0001))

    print("IoU", float(correct_pos_n) / union_n)


if __name__ == "__main__":
    testing_set = [0,5,6,11,12,17,18,22,25,28,31]
    tag = sys.argv[1]

    labels = []
    prediction = []

    for tid in testing_set:
        name = "results/" + tag + "_ret_%d.json" % tid 
        data = json.load(open(name, "r"))
        labels = labels + data[0]
        prediction = prediction + data[1]

    checkResult(labels, prediction)

    

