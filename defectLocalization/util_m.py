from xml.etree import ElementTree as ET
import untangle
import sys
import xmltodict
import numpy as np

#read in the xml data file and return an np.ndarray of [n_bugs * 4] the input data with [bug_id, bug_summary, bug_description, fixed file]
def get_data(path):
    with open(path,'r') as f:
        doc = xmltodict.parse(f.read())
        return doc

def get_single_record(record,in_list):
    info = record['buginformation']
    bug_id=record['@id']
    fixed_files = record['fixedFiles']['file']
    if isinstance(fixed_files, basestring):
       in_list.append([bug_id,info['summary'],info['description'],fixed_files])
    else:
       for f in fixed_files:
           in_list.append([bug_id,info['summary'],info['description'],f])
    return 0

def get_records(path):
    all_records = []
    doc =  get_data(path)
    [get_single_record(d,all_records) for d in doc['bugrepository']['bug']]
    return (np.ndarray(all_records))
if __name__=="__main__":
    all_records =  get_records(sys.argv[1])
      
    print(all_records[0])
 
