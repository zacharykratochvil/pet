class UltraSonicAccumulator:
    def __init__(self):
        self._ultrasonic_data = []

    def on_ultra(self, data):
        if data.cm != -1:
            self._ultrasonic_data.append(data.cm)

    def get_data(self):
        _temp = self._ultrasonic_data
        self._ultrasonic_data = []
        return _temp


class Accumulator:
    def __init__(self):
        self._data = []

    def on_data(self, data):    
        self._data.append(data.data)

    def get_data(self):
        _temp = self._data
        self._data = []
        return _temp