from optoICC.tools.ethernet_searcher import EthernetSearcher
from optoICC.icc1c import ICC1cBoard
from optoKummenberg import UnitType
from optoICC import WaveformShape
from time import sleep

def main():
    print("Ethernet connection example")
    mode = input("Static IP or automatic search? (A = auto, S = static): ").strip().upper()
    board = ICC1cBoard()
    if mode == 'A':
        devices = EthernetSearcher.search_ethernet_boards()
        if not devices:
            print("No ethernet boards found.")
            return

        print(f"Boards found ({len(devices)}):")
        for idx, dev in enumerate(devices):
            print(f"[{idx}] Serial: {dev.serial_number}, IP: {dev.ip}, Port: {dev.port}")

        try:
            selection = int(input("Select index of the device you want to connect: "))
            selected = devices[selection]
            board = ICC1cBoard(ip_address=selected.ip)
            print("Board connected successfully!")
        except Exception as e:
            print(f"Connection failed: {e}")

    elif mode == 'S':
        ip = input("Specify IP address of the board: ").strip()
        try:
            board = ICC1cBoard(ip_address=ip)
            print("Board connected successfully!")
        except Exception as e:
            print(f"Connection failed: {e}")
    else:
        print("Invalid input.")

    # Example usage
    board.channel[0].SignalGenerator.SetAsInput()
    board.channel[0].SignalGenerator.SetUnit(UnitType.CURRENT)
    board.channel[0].SignalGenerator.SetShape(WaveformShape.STAIRCASE)
    board.channel[0].SignalGenerator.SetAmplitude(0.15)
    board.channel[0].SignalGenerator.SetFrequency(2)
    board.channel[0].SignalGenerator.Run()
    print("Signal generator started")
    for index in range(10):
        print(".", end="")
        sleep(1)
    board.channel[0].SignalGenerator.Stop()
    print()
    print("Signal generator stopped")

if __name__ == "__main__":
    main()