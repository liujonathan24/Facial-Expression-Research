import pandas as pd

LBP = pd.read_json("./LBP_vector_dataframe.json")
ORB = pd.read_json("./ORB_vector_dataframe.json")

overall_list = []
for a in range(920): # Number of photos
    emotion1, vector_LBP = list(LBP.iloc[a])
    emotion2, vector_ORB = list(ORB.iloc[a])
    if emotion1 == emotion2: # Just checking to make sure nothing is wrong
        max_lbp = max(vector_LBP)
        vector_LBP = [a/max_lbp for a in vector_LBP]

        max_orb = max(vector_ORB)
        vector_ORB = [a / max_orb for a in vector_ORB]
        overall_list.append([emotion1, vector_LBP+vector_ORB])
    else:
        print('whut????')


print(len(overall_list[0][1]))

# C is a small constant
c = 1e-5

for a in range(920):
    abcd = overall_list[a][1]

