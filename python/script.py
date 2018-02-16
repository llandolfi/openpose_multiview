
import argparse
import os
import sys
import shutil

def main():
	exepath ="/home/lando/projects/INAIL/openpose_multiview/build/stereopose_rt"
	p = "/home/lando/projects/RAMCIP/video/"
	for nD in os.listdir(p):
		if nD.endswith("-D.avi"):
			D = os.path.join(p,nD)
			C = D[0:-6] + ".avi"
			T = D[0:-6] + ".openpose.csv"
			T3D = D[0:-6] + ".openpose.3d.csv"
			if not os.path.isfile(T):
				print "Processing",D
				cmd = exepath + " -video %s -depth_video %s -camera K1 -write_keypoint tmp -write_keypoint3D tmp3D" % (C,D)
				r = os.system(cmd)
				if r == 0:
					shutil.copyfile("tmp",T)
					shutil.copyfile("tmp3D",T3D)
				else:
					print "failed",C,D,T,"with",r
					print "cmd:",cmd
					break

if __name__ == '__main__':
	main()