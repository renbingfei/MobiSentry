'''
Author: Ashish Katlam
Modified: Bingfei Ren<delete features method/class/field_names and strings
for these features take too much db space
Descriptioin: This code extracts all the features of a given Android application.
              All the features are extracted using AndroGuard Tool
'''
from __future__ import division
from androguard.core.bytecodes.dvm import DalvikVMFormat
from androguard.core.analysis.analysis import VMAnalysis
from androguard.decompiler.decompiler import DecompilerDAD
from androguard.core.bytecodes.apk import APK
from androguard.core.analysis import analysis
from androguard.core.bytecodes import dvm
from constants import SPECIAL_STRINGS, DB_REGEX, API_CALLS, PERMISSIONS,NGRAM,NGRAM_THRE
import hashlib
import math
from collections import Counter
import cPickle as pickle


# Extract all features for a given application
def extract_features(file_path):
    result = {}
    try:
        a = APK(file_path)
        d = DalvikVMFormat(a.get_dex())
        dx = VMAnalysis(d)
        vm = dvm.DalvikVMFormat(a.get_dex())
        vmx = analysis.uVMAnalysis(vm)
        d.set_vmanalysis(dx)
        d.set_decompiler(DecompilerDAD(d, dx))
    except:
        return None

    result['android_version_code'] = a.get_androidversion_code()
    result['android_version_name'] = a.get_androidversion_name()
    result['max_sdk'] = a.get_max_sdk_version()
    result['min_sdk'] = a.get_min_sdk_version()
    result['libraries'] = a.get_libraries()
    result['filename'] = a.get_filename()
    result['target_sdk'] = a.get_target_sdk_version()
    result['md5'] = hashlib.md5(a.get_raw()).hexdigest()
    result['sha256'] = hashlib.sha256(a.get_raw()).hexdigest()
    result['permissions'] = a.get_permissions()
    result['activities'] = a.get_activities()
    result['providers'] = a.get_providers()
    result['services'] = a.get_services()
    #result['strings'] = d.get_strings()
    #result['class_names'] = [c.get_name() for c in d.get_classes()]
    #result['method_names'] = [m.get_name() for m in d.get_methods()]
    #result['field_names'] = [f.get_name() for f in d.get_fields()]
    class_names = [c.get_name() for c in d.get_classes()]
    method_names = [m.get_name() for m in d.get_methods()]
    field_names = [ f.get_name() for f in d.get_fields()]

    result['is_native_code'] = 1 if analysis.is_native_code(dx) else 0
    result['is_obfuscation'] = 1 if analysis.is_ascii_obfuscation(d) else 0
    result['is_crypto_code'] = 1 if analysis.is_crypto_code(dx) else 0
    result['is_dyn_code'] = 1 if analysis.is_dyn_code(dx) else 0
    result['is_reflection_code'] = 1 if analysis.is_reflection_code(vmx) else 0
    result['is_database'] = 1 if d.get_regex_strings(DB_REGEX) else 0

    s_list = []
    #s_list.extend(result['class_names'])
    #s_list.extend(result['method_names'])
    #s_list.extend(result['field_names'])
    s_list.extend(class_names)
    s_list.extend(method_names)
    s_list.extend(method_names)
    result['entropy_rate'] = entropy_rate(s_list)

    result['feature_vectors'] = {}

    # Search for the presence of api calls in a given apk
    result['feature_vectors']['api_calls'] = []
    for call in API_CALLS:
        status = 1 if dx.tainted_packages.search_methods(".", call, ".") else 0
        result['feature_vectors']['api_calls'].append(status)

    # Search for the presence of permissions in a given apk        
    result['feature_vectors']['permissions'] = []
    for permission in PERMISSIONS:
        status = 1 if permission in result['permissions'] else 0
        result['feature_vectors']['permissions'].append(status)

    result['feature_vectors']['special_strings'] = []
    for word in SPECIAL_STRINGS:
        status = 1 if d.get_regex_strings(word) else 0
        result['feature_vectors']['special_strings'].append(status)

    opt_seq = []
    for m in d.get_methods():
        for i in m.get_instructions():
            opt_seq.append(i.get_name())

    optngramlist = [tuple(opt_seq[i:i+NGRAM]) for i in xrange(len(opt_seq) - NGRAM)]
    optngram = Counter(optngramlist)
    optcodes = dict()
    tmpCodes = dict(optngram)
    #for k,v in optngram.iteritems():
    #    if v>=NGRAM_THRE:
            #optcodes[str(k)] = v
    #        optcodes[str(k)] = 1
    tmpCodes = sorted(tmpCodes.items(),key =lambda d:d[1],reverse=True) 
    for value in tmpCodes[:NGRAM_THRE]:
        optcodes[str(value[0])] = 1
    result['feature_vectors']['opt_codes'] = optcodes

    return result


def entropy_rate(data):
    for s in data:
        prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
        entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
        p = 1.0 / len(data)
        idealize = -1.0 * len(data) * p * math.log(p) / math.log(2.0)
        return round((abs(idealize) - entropy) / idealize, 2)


def create_vector_single(apk):
    feature_vector = []

    feature_vector.extend(apk['feature_vectors']['permissions'])
    feature_vector.extend(apk['feature_vectors']['api_calls'])
    feature_vector.extend(apk['feature_vectors']['special_strings'])
    
    #modified to remove optcodes start
    #feature_vector.extend(apk['feature_vectors']['opt_codes'])
    optFile = open('optCodes.p','rb')
    optCodes = pickle.load(optFile)
    optFile.close()
    opt_codes = []
    tmp_codes = []
    for i in xrange(len(optCodes)):
        value = 0
        if optCodes[i] in apk['feature_vectors']['opt_codes']:
            value = apk['feature_vectors']['opt_codes'][optCodes[i]]
        tmp_codes.append(value)
    # normalization values
    #max_v = max(tmp_codes)
    #min_v = min(tmp_codes)
    #for i in tmp_codes:
    #    opt_codes.append(i/(max_v-min_v))
    #feature_vector.extend(opt_codes)
    feature_vector.extend(tmp_codes)
    #modified to remove optcodes end

    
                

    entropy_rate = int(apk['entropy_rate'])
    native = int(apk['is_crypto_code'])
    db = int(apk['is_database'])
    feature_vector.append(entropy_rate)
    feature_vector.append(native)
    feature_vector.append(db)


    return feature_vector
