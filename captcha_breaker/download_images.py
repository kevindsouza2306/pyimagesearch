__author__ = 'kevin'
import argparse
import time
import requests
import os

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", help="path to output", required=True)
ap.add_argument("-n", "--numimages", help="# of images to download", type=int, default=500)
args = vars(ap.parse_args())

url = "https://www.e-zpassny.com/vector/jcaptcha.do"
total = 0

print(args["numimages"])
for i in range(0, args["numimages"]):
    try:
        r =requests.get(url, timeout=60)
        p =os.path.sep.join([args["outputimage"], "{}.jpg".format(str(total).zfill(5))])
        f = open(p, "wb")
        f.write(r.content)
        f.close()

        print("[INFO] downloaded: {}".format(p))
        total += 1

    except:
        print("[INFO] Error downloading image....")

    time.sleep(0.1)
