import signal
import socket
import sys
import time
from os import path

import numpy as np
import pyvista as pv

max_rows: int = 16_000
ip_got = "192.168.128.50"
port_got = 18002
loc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
loc_socket.connect((ip_got, port_got))
print("connected")

file = "09_dab"

out = path.join("./data/run8/", file)
if path.exists(out):
    raise(Exception("file exists already"))
fd =  open(out, 'a')

def handler(signum, frame):
    print('closing...')
    fd.close()
    loc_socket.close()
    sys.exit(0)

signal.signal(signal.SIGINT, handler)

#pl = pv.Plotter(notebook=False)
#pl.add_axes()
#pl.enable_eye_dome_lighting()
#pl.show(interactive_update=True)

buffer = b''
p = 0
for i in range(max_rows):

    # receive data till ';'
    while b';' not in buffer:
        buffer += loc_socket.recv(1024)

    line, _, buffer = buffer.partition(b';')
    line = line.decode('utf-8')
    fd.write(str(time.perf_counter_ns())+';')
    fd.writelines(line + ';\n')
    print(i)

#    if p%5 == 0:
 #       ds = line.split(',')
  #      pl.add_points(pv.pyvista_ndarray(np.array([ds[3], ds[4], ds[5]], dtype=np.float32)))
   #     pl.update()
    
   # p += 1

fd.close()
loc_socket.close()


"""
The format looks like this (receiver 1 to N):
<RealTimeStampMS>,<TxAddress>,<X>,<Y>,<Z>,<R1Address>,<R1Distance>,<R1Level>,<
...>,<...>,<...>,<RNAddress>,<RNDistance>,<RNLevel>

[i]
[0] <Time> - Milliseconds after start”
[1] <Sender ID> - The specific transmitter ID, can be seen on the label
[3] <x> - The x-coordinate in mm
[4] <y> - The y-coordinate in mm
[5] <z> - The z-coordinate in mm is 0 if 2 D scenario.
<Satellite ID> - The specific Satellite ID, can be seen on the label
<Distance> - The measured distance in mm
"""
