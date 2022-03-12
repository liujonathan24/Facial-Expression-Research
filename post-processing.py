import pandas as pd

LBP = pd.read_json("./vector_dataframe.json")
#ORB = pd.read_json("./ORB_dataframe.json")

#print(LBP.shape)
#print(list(LBP.iloc[0]))
for a in range(920): # Number of photos
    emotion1, vector_LBP = list(LBP.iloc[a])
    #emotion2, vector_ORB = list(ORB.iloc[a])
    #if emotion1 == emotion2: # Just checking to make sure nothing is wrong
    max_lbp = max(vector_LBP)
    vector_LBP = [a/max_lbp for a in vector_LBP]

    max_orb = max(vector_ORB)
    vector_LBP = [a / max_lbp for a in vector_LBP]

    print(vector_LBP)

    print(emotion1)
    #print(vector_ORB)
    break