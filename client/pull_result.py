import json
import flask
import imutils
import urllib3
import queue
import numpy as np
import cv2
import threading
# from fps import FPSCounter

http = urllib3.PoolManager(num_pools=5, headers={'User-Agent': 'urllib3'})
import push_stream

result_queue = queue.Queue()

EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]


def get_result(appId, stream):
    """
    向服务器发送请求获取表情识别结果
    :param path: url
    :return: error或表情识别结果数据
    """
    data = {
        "app": "live",
        "stream": "test"
    }
    print(data)

    # r = http.request_encode_body('POST', 'http://pengcheng.phi-ai.org:35690/result', fields=data)
    # r = http.request_encode_body('POST', 'http://bagua.phi-ai.org:46786/result', fields=data)
    r = http.request_encode_body('POST', 'http://bagua.phi-ai.org:46786/result', fields=data)

    if r.status != 200:
        print('error')
        return 'error'
    else:
        ret = json.loads(r.data)
        for i in range(len(ret)):
            if ret[str(i)][0] is None:
                continue
            res = {}
            preds = []
            for j in range(7):
                preds.append(float(ret[str(i)][j]))
            res['preds'] = preds
            res['label'] = ret[str(i)][7]
            res['fX'] = float(ret[str(i)][8])
            res['fY'] = float(ret[str(i)][9])
            res['fW'] = float(ret[str(i)][10])
            res['fH'] = float(ret[str(i)][11])
            result_queue.put(res)
        return json.loads(r.data)


def store_result(appId, stream):
    """
    不断循环获取结果，并将结果存储在队列中
    :param path: url
    """
    while True:
        result_queue.put(get_result(appId, stream))


def draw_result():
    """
    绘制表情识别算法的结果
    """
    cap = cv2.VideoCapture("rtmp://bagua.phi-ai.org:46785//live/test")
    # fps = FPSCounter("video")
    while True:
        # fps.count()
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        res = result_queue.get()
        print(result_queue.qsize())
        success, frame = cap.read()
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

        print("localtime", int(hours), int(minutes), int(seconds), int(milliseconds))
        frame = imutils.resize(frame, width=300)
        preds = res['preds']
        if preds is None:
            continue
        preds = np.array(preds)
        label = res['label']
        fX = int(res['fX'])
        fY = int(res['fY'])
        fW = int(res['fW'])
        fH = int(res['fH'])
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            # construct the label text
            text = "{}: {:.2f}%".format(emotion, prob * 100)

            # draw the label + probability bar on the canvas
            # emoji_face = feelings_faces[np.argmax(preds)]

            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5),
                          (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 2)
            cv2.putText(frame, label, (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),
                          (0, 0, 255), 2)
        cv2.imshow('your_face', frame)
        cv2.imshow("Probabilities", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def getin():
    t2 = threading.Thread(target=draw_result)
    t2.start()
    while True:
        get_result("live", "test")


if __name__ == '__main__':
    t2 = threading.Thread(target=draw_result)
    t2.start()
    while True:
        get_result("live", "test")

