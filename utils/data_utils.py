import gzip
import numpy as np
import lmdb
import csv
import base64
import pickle

def write_tsv_to_lmdb(source_path,output_path,field_name, max_size = 1099511627776,):
    """
    Handle pretrained image feature (bottom-up attention)
    :param source_path: list of source file path
    :param output_path: output file name
    :param max_size:
    :return: None
    """
    env = lmdb.open(output_path, map_size=max_size)
    txn = env.begin(write=True)
    for file in source_path:
        with open(file, "r") as tsv_file:
            reader = csv.DictReader(tsv_file, delimiter='\t', fieldnames=field_name)
            data = {}
            for item in reader:
                item['image_id'] = int(item['image_id'])
                data['image_h'] = int(item['image_h'])
                data['image_w'] = int(item['image_w'])
                data['num_boxes'] = int(item['num_boxes'])
                for field in ['boxes', 'features']:
                    data[field] = np.frombuffer(base64.b64decode(item[field]),
                                                dtype=np.float32).reshape((item['num_boxes'], -1))
                info = pickle.dump(data)
                txn.put(key=str(item["image_id"]),value=info)
    txn.commit()
    env.close()


