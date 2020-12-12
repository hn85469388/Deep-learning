# coding=utf-8
from flask import Flask
from flask import request
from flask import jsonify
import datetime
import hashlib
import pandas as pd
import logging
from AML import NLPDataset
from predict import *
from cleardata import newClearData, cutArticleIntoSentence, removeWrongName


app = Flask(__name__)
####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'hn85469388@gmail.com'  #
SALT = 'my_salt_hn85469388'             #
#########################################



AML = NLPDataset()
today = str(datetime.date.today())
FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, filename='myLog' + today + '.log', filemode='a', format=FORMAT)

def generate_server_uuid(input_string):
    """ Create your own server_uuid
    @param input_string (str): information to be encoded as server_uuid
    @returns server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string+SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid

def AML_PERSON(lineList):
    nameListOutPut = list()
    for Line in lineList:
        result = model.evaluate_line(sess, input_from_line(Line, FLAGS.max_seq_len, tag_to_id), id_to_tag)
        for I in range(len(result['entities'])):
            name = result['entities'][I]["word"]
            if name not in nameListOutPut:
                name.encode("utf-8")
                nameListOutPut.append(name)
    if(len(nameListOutPut)==0):
        rename = nameListOutPut
    else:
        rename = removeWrongName(nameListOutPut)
    return rename

def predict(article):
    """ Predict your model result
    @param article (str): a news article
    @returns prediction (list): a list of name
    """

    ####### PUT YOUR MODEL INFERENCING CODE HERE #######
    AML_TRUE = None
    prediction = list()

    AML_DATA = newClearData(article)  # 原始 DATA 將不需要字元移除
    logging.info(AML_DATA)
    AML_TRUE = AML.modelPredictions(AML_DATA)  # 判斷是否AML
    logging.info(AML_TRUE)

    if 1 in AML_TRUE:
        print(AML_TRUE)
        AML_person_article = cutArticleIntoSentence(AML_DATA)
        logging.info(AML_person_article)
        prediction = AML_PERSON(AML_person_article)
        logging.info(prediction)
        print(prediction)
    else:
        logging.info(prediction)
        print(AML_TRUE)
        print(prediction)
    ####################################################
    prediction = _check_datatype_to_list(prediction)
    return prediction

def _check_datatype_to_list(prediction):
    """ Check if your prediction is in list type or not. 
        And then convert your prediction to list type or raise error.
        
    @param prediction (list / numpy array / pandas DataFrame): your prediction
    @returns prediction (list): your prediction in list type
    """
    if isinstance(prediction, np.ndarray):
        _check_datatype_to_list(prediction.tolist())
    elif isinstance(prediction, pd.core.frame.DataFrame):
        _check_datatype_to_list(prediction.values)
    elif isinstance(prediction, list):
        return prediction
    raise ValueError('Prediction is not in list type.')

@app.route('/healthcheck', methods=['POST'])
def healthcheck():
    """ API for health check """
    data = request.get_json(force=True)  
    t = datetime.datetime.now()  
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL+ts)
    server_timestamp = t.strftime("%Y-%m-%d %H:%M:%S")
    print('esun_uuid : %s \n server_uuid : %s \n captain_email : %s \n server_timestamp : %s \n' %(data, server_uuid, CAPTAIN_EMAIL, server_timestamp))
    return jsonify({'esun_uuid': data['esun_uuid'], 'server_uuid': server_uuid, 'captain_email': CAPTAIN_EMAIL, 'server_timestamp': server_timestamp})

@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API """
    data = request.get_json(force=True)  
    esun_timestamp = data['esun_timestamp'] #自行取用
    
    t = datetime.datetime.now()  
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL+ts)
    
    try:
        logging.info(data['news'])
        print('news : %s \n' % (data['news']))
        answer = predict(data['news'])
    except:
        raise ValueError('Model error.')        
    server_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print('esun_timestamp : %s \n server_uuid : %s \n answer : %s \n server_timestamp : %s \n esun_uuid : %s \n' %(data['esun_timestamp'], server_uuid, answer, server_timestamp, data['esun_uuid']))
    return jsonify({'esun_timestamp': data['esun_timestamp'], 'server_uuid': server_uuid, 'answer': answer, 'server_timestamp': server_timestamp, 'esun_uuid': data['esun_uuid']})

if __name__ == "__main__":
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    with open(FLAGS.map_file, "rb") as f:
        tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, config, logger)
        app.run(host='0.0.0.0', port=80, debug=True)
