import pytest

codeastro_mode = False

def pytest_addoption(parser):
    parser.addoption(
        "--mode",
        action="store",
        metavar="NAME",
        help="different test modes NAME.",
    )

def pytest_runtest_setup(item):
    global codeastro_mode
    envnames = [mark.args[0] for mark in item.iter_markers(name="mode")]
    if envnames:
        if item.config.getoption("--mode") == 'codeastro':
            codeastro_mode = True

a1 = [83, 101, 99, 114, 101, 116, 32, 67, 111, 100, 101, 58, 32, 119, 101, 100, 105, 100, 105, 116, 33]
a2 = [78, 111, 32, 115, 101, 99, 114, 101, 116, 32, 99, 111, 100, 101, 32, 121, 101, 116, 46]
@pytest.hookimpl()
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if config.getoption("--mode") == 'codeastro':
        if terminalreporter._session.testsfailed == 0:
            vals = a1
        else:
            vals = a2
        output_str = "".join([chr(x) for x in vals])
        print(output_str)