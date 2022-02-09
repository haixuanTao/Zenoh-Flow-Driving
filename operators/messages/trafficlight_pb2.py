# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: messages/trafficlight.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1bmessages/trafficlight.proto\x12\rtraffic_light\"\xce\x02\n\rTrafficLights\x12\n\n\x02id\x18\x01 \x01(\x05\x12@\n\rtraffic_light\x18\x02 \x03(\x0b\x32).traffic_light.TrafficLights.TrafficLight\x12\r\n\x05image\x18\x03 \x01(\x0c\x1a\xdf\x01\n\x0cTrafficLight\x12\n\n\x02id\x18\x01 \x01(\x05\x12\r\n\x05score\x18\x02 \x01(\x02\x12\x0b\n\x03top\x18\x03 \x01(\x02\x12\r\n\x05right\x18\x04 \x01(\x02\x12\x0e\n\x06\x62ottom\x18\x05 \x01(\x02\x12\x0c\n\x04left\x18\x06 \x01(\x02\x12\x43\n\x05\x63olor\x18\x07 \x01(\x0e\x32\x34.traffic_light.TrafficLights.TrafficLight.LightColor\"5\n\nLightColor\x12\x07\n\x03RED\x10\x00\x12\n\n\x06YELLOW\x10\x01\x12\t\n\x05GREEN\x10\x02\x12\x07\n\x03OFF\x10\x03\x62\x06proto3')



_TRAFFICLIGHTS = DESCRIPTOR.message_types_by_name['TrafficLights']
_TRAFFICLIGHTS_TRAFFICLIGHT = _TRAFFICLIGHTS.nested_types_by_name['TrafficLight']
_TRAFFICLIGHTS_TRAFFICLIGHT_LIGHTCOLOR = _TRAFFICLIGHTS_TRAFFICLIGHT.enum_types_by_name['LightColor']
TrafficLights = _reflection.GeneratedProtocolMessageType('TrafficLights', (_message.Message,), {

  'TrafficLight' : _reflection.GeneratedProtocolMessageType('TrafficLight', (_message.Message,), {
    'DESCRIPTOR' : _TRAFFICLIGHTS_TRAFFICLIGHT,
    '__module__' : 'messages.trafficlight_pb2'
    # @@protoc_insertion_point(class_scope:traffic_light.TrafficLights.TrafficLight)
    })
  ,
  'DESCRIPTOR' : _TRAFFICLIGHTS,
  '__module__' : 'messages.trafficlight_pb2'
  # @@protoc_insertion_point(class_scope:traffic_light.TrafficLights)
  })
_sym_db.RegisterMessage(TrafficLights)
_sym_db.RegisterMessage(TrafficLights.TrafficLight)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _TRAFFICLIGHTS._serialized_start=47
  _TRAFFICLIGHTS._serialized_end=381
  _TRAFFICLIGHTS_TRAFFICLIGHT._serialized_start=158
  _TRAFFICLIGHTS_TRAFFICLIGHT._serialized_end=381
  _TRAFFICLIGHTS_TRAFFICLIGHT_LIGHTCOLOR._serialized_start=328
  _TRAFFICLIGHTS_TRAFFICLIGHT_LIGHTCOLOR._serialized_end=381
# @@protoc_insertion_point(module_scope)