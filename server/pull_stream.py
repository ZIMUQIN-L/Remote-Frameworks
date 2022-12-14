import tensorflow
import cv2
import json
import queue
import imutils
import numpy as np
import threading
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from flask import Flask, render_template, request
# from fps import FPSCounter

app = Flask(__name__)
stream_list = {}
dict = {}
result_dict = {}
q = queue.Queue()
t1 = None

def pull(camera_path, app, stream):
    """
    拉流并使用表情识别算法对其进行计算
    :param camera_path: 拉流的url
    """
    # q = queue.Queue()
    # dict[camera_path] = q
    global result_dict
    cap = cv2.VideoCapture(camera_path)
    success, frame = cap.read()
    # print("pull stream success<=======")
    # fps = FPSCounter("video")
    while success:
        # fps.count()
        #print("fps conut")
        #cv2.imshow('test', frame)
        #cv2.waitKey(1)
        success, frame = cap.read()
        frame = imutils.resize(frame, width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # q.put(get_emotion(gray))

        milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)

        seconds = milliseconds // 1000
        milliseconds = milliseconds % 1000
        minutes = 0
        hours = 0
        if seconds >= 60:
            minutes = seconds // 60
            seconds = seconds % 60
 
        if minutes >= 60:
            hours = minutes // 60
            minutes = minutes % 60

        # print("localtime", int(hours), int(minutes), int(seconds), int(milliseconds))
        result_dict[str(int(hours)) + " " + str(int(minutes)) + " " + str(int(seconds)) + " " + str(int(milliseconds))] = get_emotion(gray)
        print(result_dict)
    cv2.destroyAllWindows()
    cap.release()


@app.route('/index/hook/on_stream_changed', methods=['POST'])
def alter():
    """
    在受试者推流到客户端时获取流的相关信息，并开始拉流
    """
    global t1
    if t1 is not None and t1.is_alive():
        return json.dumps({
        'code': 0,
        "msg": "success"
        })
    
    res = request.json
    appNumber = res.get('app')
    streamId = res.get('stream')
    schema = res.get('schema')
    localIp = res.get('originSock').get('local_ip')
    localPort = res.get('originSock').get('local_port')
    regist = res.get('regist')
    if schema == "rtmp" and regist == True:
        # print("detect the stream======>")
        pull_path = "rtmp://" + localIp + ":" + str(localPort) + "//" + appNumber + "/" + streamId
        # print("begin to pull stream:" + pull_path)
        t1 = threading.Thread(target=pull, args=(pull_path, appNumber, streamId))
        t1.start()
        # pull(pull_path, appNumber, streamId)
    # stream_list[streamId] = pull_path
    return json.dumps({
        'code': 0,
        "msg": "success"
    })

@app.route('/result', methods=['POST'])
def process():
    appId = request.files.get('app')
    stream = request.files.get('stream')
    # res_queue = dict[appId + '\\' + stream]
    # dict[appId + '\\' + stream].clear()
    # print("receive request")
    # res = {}
    # res_queue = list(q.queue)
    # q.queue.clear()
    # for i in range(len(res_queue)):
    #     res[str(i)] = res_queue[i]
    res = {}
    for key in result_dict.keys():
        res[key] = result_dict[key]
    result_dict.clear()
    # print("its res")
    # print(res)
    return json.dumps(res)


detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = '_mini_XCEPTION.102-0.66.hdf5'
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]


def get_emotion(gray):
    # print("begin emotion calc======>")
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0:
        with graph.as_default():
            faces = sorted(faces, reverse=True,
                           key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
        # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
        # the ROI for classification via the CNN
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
    else:
        return None, None, None, None, None, None
    # print("end emotion calc<======")
    # print(fX, fY, fW, fH)
    # print(preds)
    return [str(preds[0]), str(preds[1]), str(preds[2]), str(preds[3]), str(preds[4]), str(preds[5]), str(preds[6]), label, str(fX), str(fY), str(fW), str(fH)]


if __name__ == '__main__':
    graph = tensorflow.get_default_graph()
    app.run(host="0.0.0.0",port=46786)
