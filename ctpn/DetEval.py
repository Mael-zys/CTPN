import numpy as np
import shapely
from shapely.geometry import Polygon,MultiPoint
import os
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
#%matplotlib inline

#area=[1280*720]*500
#data_num=500

Tr=0.8
Tp=0.4

def get_result_rectangle(path:str,num):
    files=os.listdir(path)
    rectangles=[[] for i in range(num)]

    for filename in files:
        fin=open(path+"/"+filename)
        file_id=int(filename[:-4][7:])-1
        for line in fin.readlines():
            points=line.strip().split(',')
            if len(points)>=8:
                points=[int(points[p]) for p in range(8)]
                rectangles[file_id].append(points)
    return rectangles


def Intersect(predicted,ground_truth):
    p_polyes=[Polygon(np.array(p).reshape(4,2)) for p in predicted]
    g_polyes=[Polygon(np.array(gt).reshape(4,2)) for gt in ground_truth]
    intersects=[p.intersection(gt) for p in p_polyes for gt in g_polyes]   
    return cascaded_union(intersects).area

def Area(poly):
    poly=np.array(poly).reshape(4,2)
    poly=Polygon(poly).convex_hull
    return poly.area

def Areaes(polyes):
    polyes=[Polygon(np.array(poly).reshape(4,2)).convex_hull for poly in polyes]
    return cascaded_union(polyes).area

def BestMatchG(predicted,ground_truth):
    if len(predicted)==0:
        return 0
    return max([2*Intersect([p],[ground_truth])/(Area(p)+Area(ground_truth)) for p in predicted])

def BestMatchD(predicted,ground_truth):
    if len(ground_truth)==0:
        return 0
    return max([2*Intersect([predicted],[gt])/(Area(predicted)+Area(gt)) for gt in ground_truth])
def Recall_Precision_Matrix(G,D):
    sigma=np.zeros([len(G),len(D)])
    tao=np.zeros([len(G),len(D)])
    for i,g in enumerate(G):
        for j,d in enumerate(D):
            intersect=Intersect([g],[d])
            sigma[i,j]=intersect/Area(g)
            tao[i,j]=intersect/Area(d)
    return sigma,tao

def MatchG(Gi,tr,sigma):
    k=0
    for s in sigma[Gi]:
        if s>tr:
            k+=1
    if k>1:
        return 1/(1+np.log(k))
    return k

def MatchD(Di,tp,tao):
    k=0
    for s in tao[:,Di].reshape(-1):
        if s>tp:
            k+=1
    if k>1:
        return 1/(1+np.log(k))
    return k

def Confusion_matrix(predicted,ground_truth,area):
    gt_area=Areaes(ground_truth)
    p_area=Areaes(predicted)
    intersect=Intersect(predicted,ground_truth)
    TP=intersect
    FP=p_area-intersect
    TN=area-gt_area-p_area+intersect
    FN=gt_area-intersect
    return TP,FP,TN,FN

def Recall(predicted,ground_truth,method="simple",average="micro",area=[]):
    result=0.0
    num=0
    if method=="simple":
        for i in range(len(ground_truth)):
            G=len(ground_truth[i])
            G_sum=0
            for gt in ground_truth[i]:
                G_sum+=BestMatchG(predicted[i],gt)
            if average=='macro':
                if G!=0:
                    result+=G_sum/G
                num+=1
            else:
                result+=G_sum
                num+=G
    elif method=="DetEval":
        for i in range(len(ground_truth)):
            G=len(ground_truth[i])
            G_sum=0
            sigma,tao=Recall_Precision_Matrix(ground_truth[i],predicted[i])
            for j in range(G):
                G_sum+=MatchG(j,Tr,sigma)
            if average=='macro':
                if G!=0:
                    result+=G_sum/G
                num+=1
            else:
                result+=G_sum
                num+=G
    else:
        i=0
        for p,gt in zip(predicted,ground_truth):
            TP,FP,TN,FN=Confusion_matrix(p,gt,area[i])
            if TP+FN!=0:
                result+=TP/(TP+FN)
            num+=1
            i+=1

    return result/num

def Precision(predicted,ground_truth,method="simple",average="micro",area=[]):
    result=0.0
    num=0
    if method=="simple":
        for i in range(len(ground_truth)):
            P=len(ground_truth[i])
            P_sum=0
            for p in predicted[i]:
                P_sum+=BestMatchD(p,ground_truth[i])
            if average=='macro':
                if P!=0:
                    result+=P_sum/P
                num+=1
            else:
                result+=P_sum
                num+=P
    elif method=="DetEval":
        for i in range(len(predicted)):
            P=len(predicted[i])
            P_sum=0
            sigma,tao=Recall_Precision_Matrix(ground_truth[i],predicted[i])
            for j in range(P):
                P_sum+=MatchD(j,Tp,tao)
            if average=='macro':
                if P!=0:
                    result+=P_sum/P
                num+=1
            else:
                result+=P_sum
                num+=P
    else:
        i=0
        for p,gt in zip(predicted,ground_truth):
            TP,FP,TN,FN=Confusion_matrix(p,gt,area[i])
            if TP+FP!=0:
                result+= TP/(TP+FP)
            num+=1
        i+=1

    return result/num

def Accuracy(predicted,ground_truth,area=[]):
    result=0.0
    num=0
    i=0
    #print(len(predicted))
    for p,gt in zip(predicted,ground_truth):
        TP,FP,TN,FN=Confusion_matrix(p,gt,area[i])
        result+=(TP+TN)/(TP+FP+FN+TN)
        #print(i)
        num+=1
        i+=1
    return result/num
def F1_score(predicted,ground_truth,area=[]):
    result=0.0
    num=0
    i=1
    for p,gt in zip(predicted,ground_truth):
        recall=Recall([p],[gt],method="DetEval",average="macro",area=area)
        precision=Precision([p],[gt],method="DetEval",average="macro",area=area)
        if recall+precision!=0:
            result+=2*(recall * precision) / (recall + precision)
        num+=1
        i+=1
    return result/num

def evaluate(predicted,ground_truth,area):
    #recall_micro=Recall(predicted,ground_truth,average="micro",area=area)
    recall_macro=Recall(predicted,ground_truth,average="macro",area=area)
    #precision_micro=Precision(predicted,ground_truth,average="micro",area=area)
    precision_macro=Precision(predicted,ground_truth,average="macro",area=area)
    #recall_deteval_micro=Recall(predicted,ground_truth,method="DetEval",average="micro",area=area)
    recall_deteval_macro=Recall(predicted,ground_truth,method="DetEval",average="macro",area=area)
    #precision_deteval_micro=Precision(predicted,ground_truth,method="DetEval",average="micro",area=area)
    precision_deteval_macro=Precision(predicted,ground_truth,method="DetEval",average="macro",area=area)
    #accuracy=Accuracy(predicted,ground_truth,area=area)
    #recall_ours=Recall(predicted,ground_truth,method="ours",area=area)
    #precision_ours=Precision(predicted,ground_truth,method="ours",area=area)
    f1_score=F1_score(predicted,ground_truth,area=area)
    #print("Recall micro:{}".format(recall_micro))
    print("Recall :{}".format(recall_macro))
    #print("Precision micro:{}".format(precision_micro))
    print("Precision :{}".format(precision_macro))
    #print("Recall DetEval micro:{}".format(recall_deteval_micro))
    print("Recall DetEval :{}".format(recall_deteval_macro))
    #print("just put the predicted result into conderation.")
    #print("Precision DetEval micro:{}".format(precision_deteval_micro))
    print("Precision DetEval :{}".format(precision_deteval_macro))
    #print("ours'method:")
    #print("Accuracy:{}".format(accuracy))
    #print("Recall:{}".format(recall_ours))
    #print("Precision:{}".format(precision_ours))
    print("F1_score :{}".format(f1_score))
    
    return f1_score
    #return recall_micro,recall_macro,precision_micro,precision_macro,recall_deteval_micro,recall_deteval_macro,precision_deteval_micro,precision_deteval_macro,accuracy,recall_ours,precision_ours,f1_score

def show_comparison(east_result,pixel_result,fots_result):
    plt.clf()
    fig = plt.figure(figsize=[40,30])
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax.bar(np.array(list(range(len(east_result)))),east_result,0.3)
    ax.bar(np.array(list(range(len(pixel_result))))+0.3,pixel_result,0.3)
    ax.bar(np.array(list(range(len(fots_result))))+0.6,fots_result,0.3)
    ax.set_xticks(np.array(list(range(len(east_result)))))
    ax.set_ylim([0,1.05])
    ax.set_xticklabels(["Recall micro","Recall macro","Precision micro","Precision macro","Recall DetEval micro","Recall DetEval macro","Precision DetEval micro","Precision DetEval macro","Accuracy","Recall","Precision","F1_score "])
    ax.legend(["east","pixel_link","FOTS"],prop={'size': 25})
    plt.plot()
    plt.savefig('./test_result.png')

def get_areas(file:str):
    with open(file,"r") as fout:
        areas=fout.readlines()
    areas=[int(a[:-1]) for a in areas]
    return areas