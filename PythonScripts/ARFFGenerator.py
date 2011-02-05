'''
Created on Oct 1, 2010

@author: marco gillies
'''

import random

num_sequences = 200

seq_ids = ["seq_"+str(i) for i in range(num_sequences)]

header = """
% 1. Title: Generated HMM dataset
% 
% 2. Sources:
%      (a) Creator: Marco Gillies

@relation randomHMM
 
"""

seq_id_attr = "@attribute sequence_id {" + ",".join(seq_ids) + "}\n"
 
mainattrs = """
@attribute bag relational
   @attribute output {0,1,2,3,4,5}
@end bag
@attribute class {0,1}
 
@data
"""

def generateData(classid):
    if classid == 0:
        output = 0
        while output < 6:
            while random.randint(0, 9) < 9:
                yield output
            output += 1
    else:
        output = 5
        while output >= 0:
            while random.randint(0, 9) < 9:
                yield output
            output -= 1

data = []
for id in seq_ids:
    classid = random.randint(0,1)
    seq = "\"" + ",".join([str(i) for i in generateData(classid)]) + "\""
    data.append(",".join([id, seq, str(classid)]))


if __name__ == '__main__':
    print header + seq_id_attr + mainattrs + "\n".join(data)