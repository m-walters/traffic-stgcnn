import logging

class logman:
    def __init__(self, name, level, sformat, logfile=None):
	level = level.upper()
	self.name = name
	self.logger = logging.getLogger(name)
	# Default level must be debug, otherwise lower level messages won't be
	# passed to handlers
	self.logger.setLevel(logging.DEBUG)
	if logfile:
	    handle = logging.FileHandler(logfile)
	else:
	    handle = logging.StreamHandler()
	handle.setLevel(getattr(logging, level))
	handle.setFormatter(logging.Formatter(sformat))
	self.logger.addHandler(handle)


    def add_handler(self, level, sformat, fname=None):
        level = level.upper()
	logging.getLogger(self.name)
        if fname:
	    handle = logging.FileHandler(fname)
	else:
	    handle = logging.StreamHandler()
	handle.setLevel(getattr(logging, level))
	handle.setFormatter(logging.Formatter(sformat))
	self.logger.addHandler(handle)


    def printl(self, level, msg):
	level = level.lower()
        getattr(self.logger, level)(msg)
