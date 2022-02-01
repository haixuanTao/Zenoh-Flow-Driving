from operators.source import MySrc
from operators.sink import MySink
from operators.operator import MyOp


class MockInput:
    def __init__(self, bytes):
        self.data = bytes


def tests_integrations():
    source = MySrc([])
    source_configuration = source.initialize([])
    source_output = source.run([], source_configuration)

    mock_operator_input = {"Data": MockInput(source_output)}
    operator = MyOp([])
    operator_configuration = operator.initialize([])
    operator_output = operator.run(
        [], operator_configuration, mock_operator_input
    )

    mock_sink_input = MockInput(operator_output.get("Data"))

    sink = MySink()
    sink_configuration = sink.initialize([])
    sink_output = sink.run([], sink_configuration, mock_sink_input)


tests_integrations()