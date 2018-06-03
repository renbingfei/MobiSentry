#coding=utf-8

from pymongo import MongoClient
from constants import DB_NAME
from core2 import extract_features
import multiprocessing
import sys
import os

DATA_TYPE = None


def call_and_add(fname):
    global DATA_TYPE
    try:
        flag = False
        f = open('record.txt','r')
        for line in f:
            if fname == line.strip('\n'):
                print '[!]'+fname
                flag = True
        f.close()
        if not flag:
            print '[*] Feature Extracting: {}'.format(fname)
            result = extract_features(fname)
            print '[*] Adding to db: {}'.format(fname)
            client = MongoClient()
            db = client[DB_NAME]
            result['data_type'] = DATA_TYPE
            db['apk'].update({'sha256': result['sha256']}, result, upsert=True)
            '''
            forbidden the delete operation
            '''
            #os.remove(fname)
            #add one record to file when apps finished their analysis
            f = open('record.txt','a')
            f.write(fname+"\n")
            f.close()
    except Exception as e:
        print '[!] Error occured with {}, {}'.format(fname, e)


def perform_analysis(d, t):
    global DATA_TYPE
    DATA_TYPE = t
    file_names = []
    index = 0
    for root, dirs, files in os.walk(d):
        for name in files:
            file_name = os.path.join(root, name)
            file_names.append(file_name)
            call_and_add(file_name)
            index += 1
            print '[','*'*40,'] already process',index

if __name__ == '__main__':
    if len(sys.argv) > 2:
        #single path
        dir_path = sys.argv[1]
        data_type = sys.argv[2]
        if data_type not in ('goodware','malware'):
            print '[+] You should use goodware or malware as data type'
        else:
            perform_analysis(dir_path,data_type)
    else:
        print '[+] Usage: python {} <dir_path> <data_type>'.format(__file__)
