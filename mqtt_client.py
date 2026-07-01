"""
mqtt_client.py
--------------------------------
MQTT Publisher
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

        try:

            self.client.connect(

                MQTT_BROKER,

                MQTT_PORT,

                60

            )

            self.client.loop_start()

            print("MQTT Connected")

        except Exception as e:

            print("MQTT Error :", e)

            self.enable = False

    # =====================================

    def publish(

        self,

        bpm,

        confidence,

        fps

    ):

        if not self.enable:
            return

        payload = {

            "bpm": round(float(bpm), 1),

            "confidence": round(float(confidence), 1),

            "fps": round(float(fps), 1)

        }

        try:

            self.client.publish(

                MQTT_TOPIC,

                json.dumps(payload)

            )

        except Exception as e:

            print(e)

    # =====================================

    def disconnect(self):

        if not self.enable:
            return

        self.client.loop_stop()

        self.client.disconnect()