import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import simpledialog
from abc import ABC, abstractmethod

from src.Parameters import Parameters



class GUI(ABC):
    """
    This class will act as a parent for more specific GUI classes.

    """
    _instances = []

    def __new__(cls, *args, **kwargs):
        cls._instances.append(super().__new__(cls))
        return cls._instances[-1]
    
    def __init__(self):
        super().__init__()
        self.params = {}

    def update_parameters(self):
        """
        This method will update the parameters of the pipeline.

        :param kwargs: dictionary of parameters
        :return:
        """
        Parameters.update_parameters(self.params)

    def consilidate_parameters(self):
        """
        This method will consilidate the parameters of the pipeline.

        :return: dictionary of parameters
        """
        params = Parameters.get_parameters()
        
        # update default params with the new params
        for key, value in params.items():
            self.defaults[key] = value

    @abstractmethod
    def create_gui(self):
        pass

    @abstractmethod
    def create_widgets(self):
        pass




class StepGUI(GUI):
    """
    This class will be used to create a GUI for an individual step.

    How it will be done is it will get the parameters required by the step and create a GUI for them.
    It will do this by creating a new window and use the tkinter library to create the widgets.
    Each widget will be created based on the type of the parameter
    Sliders will be used for integers and floats
    The sliders will have the ablility to change the range of the from the widget.
    Checkboxes will be used for booleans
    Textboxes will be used for strings
    Dropdowns will be used for enums
    Unkowns will be textboxes

    It will also create a 2 windows, one for the plots created by the step.
    The other will show the print statements from the step.

    When closed the parameters will be save and put into the Parameters class.

    """
    def __init__(self):
        super().__init__()
        self.params_to_ignore = ['self', 'kwargs', 'args', 'return', 'None', '', 'masks', 'images', 'image', 
                                 'cell_mask', 'nuc_mask', 'cyto_mask', 'nucChannel', 'FISHChannel', 'cytoChannel',
                                 'independent_params', 'timepoint', 'position']

    def create_gui(self, step):
        # get the parameters required by the step
        self.required_params, self.all_params, self.defaults, self.types = step.get_parameters()

        self.consilidate_parameters()

        self.create_field()

    def create_field(self):
        # create a new window
        self.window = tk.Tk()
        self.window.title("Step Parameters")

        # create a frame for the widgets
        self.frame = ttk.Frame(self.window)
        self.frame.pack(padx=10, pady=10)

        # create a label for the window
        ttk.Label(self.frame, text="Step Parameters").pack()

        self.create_widgets()

        # create a button to save the parameters
        ttk.Button(self.frame, text="Save", command=self.save_parameters).pack()

        self.window.mainloop()


    def create_widgets(self):
        # Remove all unwanted parameters
        for param in self.params_to_ignore:
            if param in self.all_params:
                self.all_params.remove(param)

        # create the widgets for the parameters
        self.widgets = []
        for param in self.all_params:
            if self.types[param] == int:
                self.widgets.append(self.create_slider(param))
            elif self.types[param] == float:
                self.widgets.append(self.create_slider(param))
            elif self.types[param] == bool:
                self.widgets.append(self.create_checkbox(param))
            elif self.types[param] == str:
                self.widgets.append(self.create_textbox(param))
            elif self.types[param] == list:
                self.widgets.append(self.create_dropdown(param))
            else:
                self.widgets.append(self.create_textbox(param))



    def create_slider(self, param):
        frame = ttk.Frame(self.frame)
        frame.pack(fill='x', padx=5, pady=5)

        label = ttk.Label(frame, text=param)
        label.pack(side='left')

        default = self.defaults.get(param, 1)
        slider = ttk.Scale(frame, from_=default*0.5, to=default*1.5, orient='horizontal')
        slider.set(default)
        slider.pack(side='right', fill='x', expand=True)
        return slider
    
    def create_checkbox(self, param):
        frame = ttk.Frame(self.frame)
        frame.pack(fill='x', padx=5, pady=5)

        label = ttk.Label(frame, text=param)
        label.pack(side='left')

        checkbox = ttk.Checkbutton(frame)
        checkbox.pack(side='right')
        return checkbox
    
    def create_textbox(self, param):
        frame = ttk.Frame(self.frame)
        frame.pack(fill='x', padx=5, pady=5)

        label = ttk.Label(frame, text=param)
        label.pack(side='left')

        textbox = ttk.Entry(frame)
        textbox.insert(0, self.defaults[param])
        textbox.pack(side='right', fill='x', expand=True)
        return textbox
    
    def create_dropdown(self, param):
        frame = ttk.Frame(self.frame)
        frame.pack(fill='x', padx=5, pady=5)

        label = ttk.Label(frame, text=param)
        label.pack(side='left')

        dropdown = ttk.Combobox(frame, values=self.defaults[param])
        dropdown.pack(side='right')
        return dropdown




if __name__ == '__main__':
    gui = StepGUI()
    gui.create_gui(BIGFISH_SpotDetection)







