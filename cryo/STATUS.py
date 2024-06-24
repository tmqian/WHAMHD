import usb
import usb.core

usb.core.find()

dev = usb.core.find(idVendor=0x04d8, idProduct=0xf420)

stat = [0x06,0x55,0xaa,0x00,0x00,0xe5,0xb5,0xff]

x = dev.write(0x01,stat)

print(x)

y = dev.read(0x81,64)

print(y)




