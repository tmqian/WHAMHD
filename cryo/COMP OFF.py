import usb
import usb.core

usb.core.find()

dev = usb.core.find(idVendor=0x04d8, idProduct=0xf420)

comp2 = [0x07,0x55,0xaa,0x01,0x00,0x39,0x00,0x52]

x = dev.write(0x01,comp2)

print(x)

y = dev.read(0x81,64)

print(y)



