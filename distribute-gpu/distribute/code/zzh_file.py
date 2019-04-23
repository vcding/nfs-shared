import os 
import csv
import json

txt_stop_flag = "./stop_flag.txt"
csv_file_name = "./bath_size_result.csv"
tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

role_enum = {
    "chief": "worker",
    "worker": "chief"
}
def clear_csv_file():
    try:
        file = open(csv_file_name, 'w', newline="")
        writer = csv.writer(file)
        writer.writerow(['timestap', 'global_stop', 'type', 'loss', 'accuracy', 'lasttime'])
        remove_flag()
        print("----- clear the csv file -------")
    finally:
        if file:
            file.close()

def write_csv_file(record):
    try:
        file = open(csv_file_name, 'a', newline="")
        writer = csv.writer(file)
        for i in range(len(record)):
            writer.writerow(record[i])
        print("\033[31m---- The length of record is [%d] has been write ---\033[0m" %(len(record)))
    finally:
        if file:
            file.close()
            
def get_stop_flag(role):
    return os.path.exists("./stop_flag_" + role_enum[role])

def write_stop_flag(role):
    try:
        file = open("./stop_flag_" + role, 'w')
        print("\033[31m---- The [%s] has been ready ---\033[0m" %(role))
    finally:
        if file:
            file.close()

def remove_flag():
    if os.path.exists("./stop_flag_worker"):
        os.remove("./stop_flag_worker")
    if os.path.exists("./stop_flag_chief"):
        os.remove("./stop_flag_chief")

if __name__ == "__main__":
    print(get_stop_flag(tf_config['task']['type']))
