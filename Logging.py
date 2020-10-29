"""
@package Logging.py

The class for logging the optimization procedure, e.g. screen print-out messages.

@author Daning Huang
@date   11/05/2019
"""

try:
    from mpi4py import MPI
    _RANK = MPI.COMM_WORLD.rank
except ImportError:
    _RANK = 0

class logging(object):
    """
    The logging class.
    """
    def __init__(self, lvl=None):
        self.printLvl = lvl  # Has to be specified in derived class.
        self.printOff = 0    # Offset

    @property
    def printLvl(self):
        """Level of details in print-out messages."""
        return self._printLvl

    @printLvl.setter
    def printLvl(self, val):
        self._printLvl = val

    @property
    def printOff(self):
        """Number of extra preceeding tabs."""
        return self._printOff

    @printOff.setter
    def printOff(self, val):
        self._printOff = val

    def printMsg(self, message, priority, proc=0, **kwargs):
        """
        Print out messages on the specified processor.
        @param message Message to print.
        @param priority Priority of the message, i.e. print if lvl >= prr.
        @param proc The processor to print message.
        @param kwargs Arguments for print.
        """
        if _RANK == proc:
            _t = priority + self._printOff
            if self._printLvl >= priority:
                print("    "*_t + message, **kwargs)
