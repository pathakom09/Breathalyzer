import pygatt

def receive_data():
    # Set the Bluetooth device address of your server
    server_address = '4C:75:25:A2:19:D2'  # Replace with your server's Bluetooth address

    try:
        # Connect to the Bluetooth server
        adapter = pygatt.GATTToolBackend()
        adapter.start()
        device = adapter.connect(server_address)

        print("Connected to Bluetooth device")

        while True:
            # Receive data from the Bluetooth server
            data = device.char_read("00001101-0000-1000-8000-00805f9b34fb").decode("utf-8")

            # Print the received data
            print("Received data: {}".format(data))

    except KeyboardInterrupt:
        print("Connection closed by the user.")
    except Exception as e:
        print("Error:", str(e))
    finally:
        # Stop the Bluetooth adapter
        adapter.stop()

if __name__ == "__main__":
    receive_data()
