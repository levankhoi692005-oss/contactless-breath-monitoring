"""
mqtt_client.py
"""

import json
import paho.mqtt.client as mqtt

from config import *

class MQTTClient:

    def __init__(self):

        self.enable = MQTT_ENABLE

        if not self.enable:
            return

        self.client = mqtt.Client()

        self.client.connect(

            MQTT_BROKER,

            MQTT_PORT,

            60

        )

        print("MQTT Connected")

    def publish(

            self,

            bpm,

            confidence,

            fps

    ):

        if not self.enable:
            return

        payload = {

            "bpm": float(bpm),

            "confidence": float(confidence),

            "fps": float(fps)

        }

        self.client.publish(

            MQTT_TOPIC,

            json.dumps(payload)

        )

    def disconnect(self):

        if not self.enable:
            return

        self.client.disconnect()