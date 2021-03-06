import xmltodict
import re
def get_data(path):
    with open(path,'r') as f:
        doc = xmltodict.parse(f.read())
        return doc['bugrepository']['bug']

def get_single_record(record,in_list):
    info = record['buginformation']
    clean_info = info['description'].replace("."," ")
    clean_info = clean_info.replace("/"," ")
    clean_info = clean_info.replace("_"," ")

    clean_info = re.sub(r"\d+" ,"",clean_info)
    clean_info = re.sub(r"[^\W]" ," ",clean_info)
    pattern = re.compile(r'[a-z][A-Z]')
    cat = pattern.search(clean_info)
    while(cat is not None):
        i = cat.start()+1
        clean_info = clean_info[:i]+ " " +clean_info[i:]
        cat = pattern.search(clean_info)
    bug_id=record['@id']
    fixed_files = record['fixedFiles']['file']
    if isinstance(fixed_files, basestring):
       splited_file = fixed_files.replace('/','.')
       splited_file = splited_file.split('.')
       if len(splited_file)>2:
           in_list.append([bug_id,info['summary'],clean_info,[".".join(splited_file[:-2])]])
    else:
       splited_files = []
       for f in fixed_files:
           splited_file = f.replace('/','.')
           splited_file = splited_file.split('.')
           if len(splited_file)>2:
               splited_files.append(".".join(splited_file[:-2]))
          
       in_list.append([bug_id,info['summary'],clean_info,splited_files])
    return 0
def get_flat_record(record, in_list):

#  info = record['buginformation']
#   bug_id=record['@id']
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
    [get_single_record(d,all_records) for d in doc if d['buginformation']['description'] is not None]
    return all_records


def precision_score(y_true, y_predict):
    total_selected = len(y_predict)
    hit_elements = [i for i in y_predict if i in y_true]
    hit = float(len(hit_elements))
    return (hit/total_selected)


def recall_score(y_true, y_predict):
    total_relevant = len(y_true)
    hit_elements = [i for i in y_predict if i in y_true]
    hit = float(len(hit_elements))
    return (hit/total_relevant)


def hit_or_not(y_true,y_predict):
    hit = 0
    for i in y_predict:
        if i in y_true:
            hit = 1
            break
    return (hit)





if __name__ == "__main__":
    path = "/home/zhuyuecai/workspace/AITour/defectLocalization/data/EclipseBugRepository.xml"
    all_records = []
    doc =  get_data(path)
    #[get_single_record(d,all_records) for d in doc]
    #print(all_records[0:3])
    #print("====================================")
    #all_records = []
    [get_flat_record(d,all_records) for d in doc]
    le = []
    bugs = []
    for r in all_records:

       ss = r[3].split('.')
       if len(ss) > 1:
           bugs.append(".".join(ss[:-1]))
       #le.append(len( r[3].split('.')))

    #print(max(le))
    #print(min(le))
    
    print(len(bugs))
    sbug = set(bugs)
    print(len(sbug))




