from multi import plot9, readLog

import sys
shot = int(sys.argv[1])
#plot9(shot)

try:
    log = readLog(sys.argv[2])
    entry = log[shot]
    tag = f"{shot} : {entry}"
    plot9(shot, tag = tag)
    print(f"{shot} log detected")

except:
    plot9(shot)

