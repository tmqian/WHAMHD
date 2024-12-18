# update 12/9/2024

import usb.core
import usb.util

import sys

# Find the device
dev_list = list(usb.core.find(idVendor=0x04d8, idProduct=0xf420, find_all=1))

try:
    idx = int(sys.argv[1])
except:
    idx = 0
dev = dev_list[idx]

stat = [0x06,0x55,0xaa,0x00,0x00,0xe5,0xb5,0xff]

if dev:
    try:
        # Detach the kernel driver if active
        if dev.is_kernel_driver_active(0):
            print("Detaching kernel driver...")
            dev.detach_kernel_driver(0)

        # Set the active configuration
        dev.set_configuration()
        cfg = dev.get_active_configuration()
        print("Active configuration set:") 
        #print("Active configuration set:", cfg)

        # Claim the interface
        usb.util.claim_interface(dev, 0)

        # Write data to the endpoint
        x = dev.write(0x01, stat)
        y = dev.read(0x81,64)

        print(f"    Success! Received {y[0]} bytes")

    except Exception as e:
        print("An error occurred:", e)
    finally:
        # Always release the interface
        try:
            usb.util.release_interface(dev, 0)
            print("Interface released.")
        except Exception as e:
            print("Could not release interface:", e)
else:
    print("Device not found.")
