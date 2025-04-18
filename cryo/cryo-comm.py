'''
This code talks to the Trillium M600 compressors

T. Qian - 9 Dec 2024
'''

import usb.core
import usb.util

import sys
from array import array

# The byte array
test_data = array('B', [26, 85, 170, 20, 0, 10, 172, 251, 29, 0, 88, 1, 89, 95, 0, 0, 0, 0, 0, 0, 0, 0, 211, 251, 29, 0, 248, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255])


# Descriptions for each bit in the states
digital_output_descriptions = [
    "Compressor Pressure Fault", "Compressor Temperature Fault", "Phase Error Fault",
    "System Running", "Fan/Heater On", "Compressor/Solenoid On", "Coldhead On",
    "Spare Output #1 On", "Spare Output #2 On"
]

digital_input_descriptions = [
    "Coldhead Enable", "Compressor/Solenoid Enable", "System Reset",
    "Compressor Pressure Fault", "Compressor Temperature Fault", "Compressor Overload Fault",
    "Spare Remote Input #1", "Spare Remote Input #2", "Spare Chassis Input #1"
]

button_descriptions = [
    "On Button Pressed", "Off Button Pressed", "Menu Button Pressed",
    "History Button Pressed"
]

fault_descriptions = [
    "System Fault", "AC 220V Fuse Blown", "AC 24V Fuse Blown",
    "Compressor Overload Fault", "AC Phase Fault", "Compressor Pressure Fault",
    "Compressor Temperature Fault", "3 Phases Detected; 1 Phase Expected",
    "1 Phase Detected; 3 Phases Expected"
]

class CryoComm:

    def __init__(self):


        for idx in range(3):

            try:
                print(f"Attempting to load data from USB connection (idx={idx})")
                self.get_status(idx = idx)
            except:
                print("    error")
                #print("    Loading test data")
                #self.data = test_data
                #self.parse()

            print("\n=====\n")

    def connect(self, dev):
        # this function is not yet used
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
                self.data = y

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

    def get_status(self, idx=0):

        stat = [0x06,0x55,0xaa,0x00,0x00,0xe5,0xb5,0xff]

        dev_list = list(usb.core.find(idVendor=0x04d8, idProduct=0xf420, find_all=1))
        dev = dev_list[idx]

        #connect(dev) # test

        print(f"    Sending 7 byte GET STATUS request...")
        x = dev.write(0x01,stat)
        y = dev.read(0x81,64)

        self.data = y
        print(f"    Success! Received {y[0]} bytes")
        self.parse()

        self.dev_list = dev_list
        self.dev = dev

        # reset device handle
        dev.reset()


    def parse(self, skip=6, show=True):

        # Extract the payload from the byte array
        payload = self.data[skip:skip+16]

        # Helper function to read 2 bytes and convert to integer
        def read_uint16(data, index):
            return data[index] + (data[index+1] << 8)

        # Helper function to read 4 bytes and convert to integer
        def read_uint32(data, index):
            return data[index] + (data[index+1] << 8) + (data[index+2] << 16) + (data[index+3] << 24)

        # Parse the payload
        self.compressor_run_time = read_uint32(payload, 0)  # 4 bytes
        self.pcb_temperature = read_uint16(payload, 4) / 10.0  # 2 bytes, 0.1 C units
        self.input_voltage = read_uint16(payload, 6)  # 2 bytes, mV

        self.digital_output_states = read_uint16(payload, 8)  # 2 bytes, bit set
        self.digital_input_states = read_uint16(payload, 10)  # 2 bytes, bit set
        self.button_states = read_uint16(payload, 12)  # 2 bytes, bit set
        self.fault_states = read_uint16(payload, 14)  # 2 bytes, bit set

        if show:
            self.status()

    def status(self):

        # Print parsed values
        print(f"\nCompressor Run Time: {self.compressor_run_time/3600:.1f} hours")
        print(f"PCB Temperature: {self.pcb_temperature} °C")
        print(f"24V Input Voltage: {self.input_voltage/1000} V")

        print(f"\nDigital Output States: {bin(self.digital_output_states)}")
        print(f"Digital Input States: {bin(self.digital_input_states)}")
        print(f"Button States: {bin(self.button_states)}")
        print(f"Fault States: {bin(self.fault_states)}")

    def states(self):
        # Function to print the bit states
        def print_bit_states(label, bit_states, descriptions):
            print(f"{label}:")
            for i in range(len(descriptions)):
                state = "ON" if bit_states & (1 << i) else "OFF"
                print(f"  Bit {i}: {descriptions[i]} - {state}")

        # Print the bit states
        print_bit_states("\nDigital Output States", self.digital_output_states, digital_output_descriptions)
        print_bit_states("Digital Input States", self.digital_input_states, digital_input_descriptions)
        print_bit_states("Button States", self.button_states, button_descriptions)
        print_bit_states("Fault States", self.fault_states, fault_descriptions)


if __name__ == "__main__":

    cryo = CryoComm()
    #cryo.states()

