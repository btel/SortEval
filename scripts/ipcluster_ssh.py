#!/usr/bin/env python
#coding=utf-8

import subprocess
import time
import os

SERVER="compute3"
N_ENGINES=2

if __name__ == "__main__":
    controller = subprocess.Popen("ipcontroller-2.6")
    time.sleep(1)
    furl_file = os.path.expanduser(
        os.path.join("~/.ipython/security/ipcontroller-engine.furl")
        )
    furl_dest = SERVER+":.ipython/security/"

    subprocess.call(["scp", furl_file, furl_dest])

    engines = []
    print "Starting engines"
    for i in range(N_ENGINES):
        engines.append(subprocess.Popen(['ssh', SERVER, 'ipengine'],
                                        stderr=subprocess.PIPE))
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        subprocess.call(['ssh',SERVER, 'killall', 'ipengine'])
        controller.terminate()
        print "Killing engines"
    print "Exitting"
        
                        


