import usb.core
import usb.util
import struct

'''
See section 6.3.9 in
https://trilliumus.com/wp-content/uploads/2017/05/trillium-m600-400-helium-compressor-manual.pdf
'''


# Endpoint configuration
ENDPOINT_OUT = 0x01  # Write endpoint
ENDPOINT_IN = 0x81  # Read endpoint

# messages
get_status = [0x06, 0x55, 0xaa, 0x00, 0x00, 0xe5, 0xb5, 0xff]

# find our device
dev = usb.core.find(idVendor=0x04d8, idProduct=0xf420)
# usb.core.find() # use this to find devices from interactive python

if dev.is_kernel_driver_active(0):
    dev.detach_kernel_driver(0)

# set active configuration
dev.set_configuration()

# Claim interface
usb.util.claim_interface(dev, 0)

## get an endpoint instance
#cfg = dev.get_active_configuration()
#intf = cfg[(0,0)]
#
#func = lambda e: \
#        usb.util.endpoint_direction(e.bEndpointAddress) == \
#        usb.util.ENDPOINT_OUT
#ep = usb.util.find_descriptor(intf, custom_match=func)



import pdb
pdb.set_trace()

# Packet header
HEADER_1 = 0x55
HEADER_2 = 0xAA

# Command ID for Get Status command
GET_STATUS_COMMAND_ID = 229

def construct_packet(command_id):
    # Construct packet in little-endian format
    packet = struct.pack("<BBH", HEADER_1, HEADER_2, 4)  # Header and packet length (4 bytes)
    packet += struct.pack("B", command_id)  # Command ID
    packet += struct.pack("<I", 0)  # Payload data (none for Get Status command)

    # Calculate CRC (dummy calculation for demonstration)
    crc = sum(packet) % 256  # Sum of all bytes modulo 256
    packet += struct.pack("B", crc)  # Append CRC byte

    return packet

# Construct Get Status command packet
get_status_packet = construct_packet(GET_STATUS_COMMAND_ID)

GET_STATUS_COMMAND = b"\xE5"
dev.write(ENDPOINT_OUT, GET_STATUS_COMMAND)
response = dev.read(ENDPOINT_IN, 64) 

print(response)
msg = b"Hello, OXI-1000!"

