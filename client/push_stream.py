import cv2 as cv
import cv2
import subprocess as sp
import shlex
import queue


def push(rtmpURL, capId):
    """
    向ZLMediaKit进行推流
    :param rtmpURL: 推流的url
    :param capId: 摄像头id，一般为0
    :param q: 存放帧的队列
    """
    cap = cv.VideoCapture(capId)
    # cap.set(cv2.CAP_PROP_FPS, FPS)
    fps = int(cap.get(cv.CAP_PROP_FPS))

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # ffmpeg command
    command = ['ffmpeg',
               '-y',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-pix_fmt', 'bgr24',
               '-s', "{}x{}".format(width, height),
               '-r', str(fps),
               '-i', '-',
               '-c:v', 'libx264',
               '-vf', 'drawtext=fontsize=60:text=\'%{pts\:hms}\'',
               # '-vf', 'drawtext=fontsize=160:text=\'%{pts: hms}\'',
               # '-vf', '\"drawtext=expansion=strftime:basetime=$(date +%s -d \'2020-11-24 16:27:50\')000000 :text=\'%Y-%m-%d %H\\:%M\\:%S\':fontsize=18:fontcolor=white:box=1:x=100:y=100:boxcolor=black@0.5:\"',
               '-pix_fmt', 'yuv420p',
               '-preset', 'ultrafast',
               '-f', 'flv',
               rtmpURL]

    # 管道配置
    # https://www.cnblogs.com/lb-blogs/p/15240638.html
    print(command)
    p = sp.Popen(command, stdin=sp.PIPE)

    # read webcamera
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Opening camera is failed")
            break

        # write to pipe
        p.stdin.write(frame.tostring())


if __name__ == "__main__":
    rtmpUrl = "rtmp://bagua.phi-ai.org:46785//live/test"
    push(rtmpUrl, 0)
