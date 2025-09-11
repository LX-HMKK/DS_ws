import serial
import time
import time
import binascii

class SerialPort:
    def __init__(self,  port: str,  baudrate: int = 1115200, timeout: int =1,
                 send_format: str = 'str', recv_format: str = 'str'):
        self.ser = None
        self.default_send_format = send_format
        self.default_recv_format = recv_format

        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=timeout
            )
            print(f"serial connect:{self.ser}")
            print(f"fault send format:{send_format},fault recv format:{recv_format}")
        except Exception as e:
            print(f"serial connect fail:{e}")
            raise ConnectionError("Serialn port initialization failed")
        
    def __del__(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("serial connect close")

    def send_data(self, data, data_format: str=None):
        if not self.ser or not self.ser.is_open:
            raise ConnectionError("Serial port not available")
        
        if data_format is None:
            data_format = self.default_send_format

        try:
            if data_format == 'str':
                encoded_data = data.encode('utf-8')
            elif data_format == 'str':
                hex_str = data.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
                if len(hex_str) % 2 != 0:
                    raise ValueError("Hex string must have even number of characters")
                encoded_data = binascii.unhexlify(hex_str)
            elif data_format == 'bytes':
                if not isinstance(data, bytes):
                    raise TypeError("For 'bytes' format, data must be bytes type")
                encoded_data = data
            elif data_format == 'ascii':
                encoded_data = data.encode('ascii')
            else:
                raise ValueError(f"Unsupported data format: {data_format}")
        
            bytes_sent = self.ser.write(encoded_data)
            self.ser.flush()  
 
            if data_format == 'hex':
                print(f"send success (hex): {data} -> {encoded_data.hex()} ({bytes_sent} bytes)")
            elif data_format == 'bytes':
                print(f"send success (bytes): {encoded_data.hex()} ({bytes_sent} bytes)")
            elif data_format == 'ascii':
                print(f"send success(ascii): {data} ({bytes_sent} bytes)")
            else:
                print(f"send success (str): {data} ({bytes_sent} bytes)")
                
            return bytes_sent
        except Exception as e:
            print(f"send success: {e}")
            raise

    def read_data(self, size: int = None, data_format: str = None):
        if not self.ser or not self.ser.is_open:
            raise ConnectionError("Serial port not available")

        if data_format is None:
            data_format = self.default_recv_format
        
        try:
            if size is None:
                data = self.ser.read_all()
            else:
                data = self.ser.read(size)
            
            if not data:
                return "" if data_format == 'str' else b"" if data_format == 'bytes' else ""
            
            if data_format == 'str':
                decoded = data.decode('utf-8', errors='replace')
                print(f"recv data (str): {decoded}")
                return decoded
            elif data_format == 'hex':
                hex_str = data.hex()
                formatted_hex = ' '.join(hex_str[i:i+2] for i in range(0, len(hex_str), 2))
                print(f"recv data (hex): {formatted_hex}")
                return formatted_hex
            elif data_format == 'bytes':
                print(f"recv data (bytes): {data.hex()}")
                return data
            elif data_format == 'ascii':
                ascii_str = data.decode('ascii', errors='replace')
                print(f"recv data(ascii): {ascii_str}")
                return ascii_str
            else:
                raise ValueError(f"Unsupported data format: {data_format}")
        except Exception as e:
            print(f"recv fail: {e}")
            raise

    def set_formats(self, send_format: str = None, recv_format: str = None):
        if send_format:
            if send_format in ('str', 'hex', 'bytes'):
                self.default_send_format = send_format
                print(f"set default send format : {send_format}")
            else:
                print(f"useless send format: {send_format},set default")
        
        if recv_format:
            if recv_format in ('str', 'hex', 'bytes'):
                self.default_recv_format = recv_format
                print(f"set default recv format: {recv_format}")
            else:
                print(f"useless recv format: {recv_format},set default")

if __name__ == '__main__':


    # send_fmt = input("set send format (str/hex/bytes, default str): ") or 'str'
    # recv_fmt = input("set recv format (str/hex/bytes, default str): ") or 'str'

    # serial_port = SerialPort(
    #     port='/dev/ttyAMA0', 
    #     baudrate=115200,
    #     send_format=send_fmt,  
    #     recv_format=recv_fmt  
    # )
    serial_port = SerialPort(
        port='/dev/ttyAMA0', 
        baudrate=115200,
        send_format="ascii",  
        recv_format="ascii"  
    )
    # arr = [1, 2, 3, 4, 5, 6, 7]
    try:
        while True:

            # if serial_port.default_send_format == 'str':
            #     send_data = input(f"send what ({serial_port.default_send_format}): ")
            # elif serial_port.default_send_format == 'hex':
            #     send_data = input(f"send what in hex ({serial_port.default_send_format}): ")
            # elif serial_port.default_send_format == 'ascii':
            #     send_data = input(f"send what ({serial_port.default_send_format}): ")
            # else:  # bytes
            #     hex_data = input(f"send what in hex ({serial_port.default_send_format}): ")
            #     try:
            #         hex_str = hex_data.replace(" ", "")
            #         if len(hex_str) % 2 != 0:
            #             print("error: hex must be even num")
            #             continue
            #         send_data = bytes.fromhex(hex_str)
            #     except ValueError as e:
            #         print(f"error: {e}")
            #         continue
            try:
                # bytes_sent = serial_port.send_data(send_data)
                # time.sleep(1)
                # print("waiting for data...")
                received = serial_port.read_data()               
                # print(f"send: {send_data} | recv: {received}")
                if received:
                    print(f" recv: {received}")
            except Exception as e:
                print(f"s&r error: {e}")

    except KeyboardInterrupt:
        print("\n end")
    except Exception as e:
        print(f"error: {e}")
