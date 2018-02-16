import json
import csv


coco = dict([(0,  "Nose"),
        (1,  "Neck"),
        (2,  "RShoulder"),
        (3,  "RElbow"),
        (4,  "RWrist"),
        (5,  "LShoulder"),
        (6,  "LElbow"),
        (7,  "LWrist"),
        (8,  "RHip"),
        (9,  "RKnee"),
        (10, "RAnkle"),
        (11, "LHip"),
        (12, "LKnee"),
        (13, "LAnkle"),
        (14, "REye"),
        (15, "LEye"),
        (16, "REar"),
        (17, "LEar")])

icoco = dict([(v,k) for k,v in coco.iteritems()])

# complete this
# make special case with if
outfield = dict([ ("Neck","neck"), ("LShoulder","left_shoulder")])

outfieldi = []
for i in range(0,17):
    coconame = coco[i]
    if coconame in outfield:
        outfieldi.append(outfield[coconame])
    else:
        outfieldi.append(None) # keep empty


def main():
    # add option to export csv with the native names
    
    nitejoint2jointidx = None
    injson = json.load(open(sys.argv[1],"rb"))

    f = ["camera","frame","subject"]
    for i in range(0,18):
        f.append("p%dx" % i)
        f.append("p%dy" % i)
        f.append("p%dconf" % i)
        
    out = csv.DictWriter(open(sys.argv[1]+".csv","wb"),delimiter=" ",fields=f)
    out.writeheader()

    for i, frame in enumerate(injson):
        for s in range(0,len(frame["skeletons"])):
            row = dict()
            row["camera"] = 0
            row["frame"] = s
            row["subject"] = 0
            # for each output field, pick input field of Nite Skeleton
            for j,nitename in enumerate(outfieldi): # 18
                if nitejoint2jointidx is None:
                    nitejoint2jointidx = dict([(joint["name"],idx) for idx,joint in enumerate(frame["skeletons"][s]["joints"])]
                if fi is None:
                    x = 0
                    y = 0
                    conf = 0
                else:
                    # check if nitename is a special case => special handling like combination
                    nj = frame["skeletons"][s]["joints"][nitejoint2jointidx[nitename]]
                    x = nj["position"]["x"]
                    y = nj["position"]["y"]
                    z = nj["position"]["z"]
                    conf = nj["confidence"]
                row["p%dx" %j] = x
                row["p%dy" %j] = y
                row["p%dconf" %j] = conf

            out.writerorw(out)

if __name__ == '__main__':
    main()