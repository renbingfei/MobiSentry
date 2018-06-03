'''
Author: Ashish Katlam
Description: Fetches data from database and create a csv file
'''
from pymongo import MongoClient
from constants import DB_NAME,TNGRAM_THRE
from core2 import create_vector_single
from collections import *
import cPickle as pickle


client = MongoClient()
db = client[DB_NAME]

permissions = []
apis = []
# Create opt_codes
optCodes = []
# Get unique permissions
with open('unique_permissions.txt','r') as fp1:
    for line in fp1:
        permissions.append(line.strip().replace(",","").replace("'",""))

# Get apis
with open('api.txt','r') as fp2:
    for line in fp2:
        apis.append(line.strip().replace(",","").replace("'",""))
#modified to remove optcodes start
#Get optCodes
mapngram = defaultdict(Counter)
index = 0
for apk in db.apk.find():
    print "index:",index
    index += 1
    ops = apk['feature_vectors']['opt_codes']
    mapngram[apk['sha256']] = Counter(ops)

print 'start to generate selected features'
cc = Counter([])
index = 0
if TNGRAM_THRE >=1:
    for d in mapngram.values():
        print 'cc:',index
        index += 1
        cc += d
selectedfeatures = dict()
tmpfeatures = dict(cc)
tmpfeatures = sorted(tmpfeatures.items(),key=lambda d:d[1],reverse=True)
print 'end to generate selected feature'
for value in tmpfeatures[:TNGRAM_THRE]:
    selectedfeatures[value[0]] = value[1]
#for k,v in cc.iteritems():
#    if v >=TNGRAM_THRE:
#        selectedfeatures[k] = v
print 'generate opt_codes to opt_codes.txt'
with open('opt_codes.txt','w') as fp3:
    for k,v in selectedfeatures.iteritems():
        fp3.write(k+",\n")
        optCodes.append(k)
print 'generate opt_codes end. start to dump optCodes'
optFile = open('optCodes.p','wb')
pickle.dump(optCodes,optFile,True)
optFile.close()

features = permissions + apis + optCodes
#modified to remove optcodes end

#features = permissions + apis
features.append('com.metasploit.stage.PayloadTrustManager')
features.append('entropy_rate')
features.append('native')
features.append('db')
features.append('class')

with open('data.csv','w+') as op:
    header = ""
    for f in features:
        header+= f.strip().replace('"','')+','
    header = header[:-1]
    op.write(header+'\n')

        index = 0
    for apk in db.apk.find():
                print 'index:',index
                index += 1
        feature_vector = create_vector_single(apk)
        str_to_write = ""
        for i,feature in enumerate(feature_vector):
            if i < len(feature_vector)-1:
                str_to_write+=str(feature)+','
            else:
                class_label = 1 if apk['data_type'] == 'malware' else 0
                str_to_write+=str(feature)+','+str(class_label)
        op.write(str_to_write+'\n')




