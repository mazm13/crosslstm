import os

def Log(modelfile, text):
	logfile = "" + modelfile[13:-3]
	logfile = "./logs/" + logfile + 'log'
	with open(logfile, 'a') as fd:
		fd.write("%s\n" % text)
	print(text)
