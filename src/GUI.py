import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import simpledialog
from abc import ABC, abstractmethod






class GUI(ABC):
    """
    This class will act as a parent for more specific GUI classes.

    """
    _instances = []

    def __new__(cls, *args, **kwargs):
        cls._instances.append(super().__new__(cls))
        return cls._instances[-1]

    def update_parameters(self):
        """
        This method will update the parameters of the pipeline.

        :param kwargs: dictionary of parameters
        :return:
        """
        


























