import xmltodict
def get_data(path):
    with open(path,'r') as f:
        doc = xmltodict.parse(f.read())
        return doc['bugrepository']['bug']

def get_single_record(record,in_list):
    info = record['buginformation']
    bug_id=record['@id']
    fixed_files = record['fixedFiles']['file']
    if isinstance(fixed_files, basestring):
       in_list.append([bug_id,info['summary'],info['description'],[fixed_files]])
    else:
       #for f in fixed_files:
       in_list.append([bug_id,info['summary'],info['description'],fixed_files])
    return 0
def get_flat_record(record, in_list):

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
    [get_single_record(d,all_records) for d in doc]
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
    path = "/home/zhuyuecai/workspace/AITour/defectLocalization/data/ZXingBugRepository.xml"
    all_records = []
    doc =  get_data(path)
    [get_single_record(d,all_records) for d in doc]
    print(all_records[0:3])
    print("====================================")
    all_records = []
    [get_flat_record(d,all_records) for d in doc]
    print(all_records[0:3])


