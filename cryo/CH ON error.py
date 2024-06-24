import usb
import usb.core

usb.core.find()

dev = usb.core.find(idVendor=0x04d8, idProduct=0xf420)

ch1 = [0x07,0x55,0xaa,0x01,0x00,0x3a,0x01,0x6d]

x = dev.write(0x01,ch1)

print(x)


y = dev.read(0x81,64)

print(y)


