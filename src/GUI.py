import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import simpledialog
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

from src.Parameters import Parameters
from src.SequentialSteps import BIGFISH_SpotDetection



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
        Parameters.validate()
        for widget, param in zip(self.widgets, self.all_params):
            if isinstance(widget, ttk.Checkbutton):
                self.params[param] = widget.instate(['selected'])
            elif isinstance(widget, ttk.Entry):
                value = widget.get()
                # check if the value is a number and evaluate it
                try:
                    value = eval(value)
                except:
                    pass
                # check if value is a string and if so split it based on spaces into a list
                if isinstance(value, str):
                    value = value.split()
                # check if value is a string and if so split it based on commas into a list
                if isinstance(value, str):
                    value = value.split(',')
                self.params[param] = value
            elif isinstance(widget, ttk.Combobox):
                self.params[param] = widget.get()
            elif isinstance(widget, ttk.Scale):
                self.params[param] = widget.get()
            
        Parameters.update_parameters(self.params)

    def consilidate_parameters(self):
        """
        This method will consilidate the parameters of the pipeline.

        :return: dictionary of parameters
        """
        Parameters.validate()
        params = Parameters.get_parameters()
        
        # update default params with the new params
        for key, value in params.items():
            self.defaults[key] = value

    def get_current_figures(self):
        """
        This method will get the current figures.

        :return: list of figures
        """
        self.fignum = plt.get_fignums()
    
    def close_created_figures(self):
        """
        This method will close the created figures.

        :return:
        """
        for fignum in plt.get_fignums():
            if fignum not in self.fignum:
                plt.close(fignum)

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
                                 'independent_params', 'timepoint', 'position', 'fov', 'verbose', 'display_plots',]

    def create_gui(self, step):
        self.step = step
        # get the parameters required by the step
        self.required_params, self.all_params, self.defaults, self.types = step.get_parameters()

        self.consilidate_parameters()
        self.get_current_figures()
        self.create_field()

    def create_field(self):
        """
        Creates a new window titled "Step Parameters" with a frame containing widgets for user interaction.
        The window includes:
        - A "Run" button to execute the step.
        - An "Update Ranges" button to update the ranges of all scale widgets based on user input.
        - A "Save" button to save the parameters.
        The function also initializes the widgets and sets up the main event loop for the window.
        """
        # create a new window
        self.window = tk.Tk()
        self.window.title("Step Parameters")

        # make plt show the plots in a new window
        plt.ion()

        # create a frame for the widgets
        self.frame = ttk.Frame(self.window)
        self.frame.pack(padx=10, pady=10)

        self.create_widgets()

        # Create a single update button next to the save button
        def update_all_ranges():
            for widget in self.widgets:
                if isinstance(widget, ttk.Scale):
                    range_frame = widget.master.master.winfo_children()[widget.master.master.winfo_children().index(widget.master) + 1]
                    children = range_frame.winfo_children()
                    if len(children) >= 4:
                        min_textbox = children[1]
                        max_textbox = children[3]
                    else:
                        messagebox.showerror("Error", "Range frame does not have enough children.")
                    try:
                        min_val = float(min_textbox.get())
                        max_val = float(max_textbox.get())
                        widget.config(from_=min_val, to=max_val)
                    except ValueError:
                        messagebox.showerror("Invalid input", "Please enter valid numbers for min and max range.")

        def run_step():
            self.step.run(0, 0)

        self.run_button = ttk.Button(self.frame, text="Run", command=run_step)
        self.run_button.pack()

        self.update_button = ttk.Button(self.frame, text="Update Ranges", command=update_all_ranges)
        self.update_button.pack()

        # create a button to save the parameters
        ttk.Button(self.frame, text="Save", command=self.update_parameters).pack()

        # add dox string to the window
        doc_string = self.step.__doc__
        if doc_string:
            ttk.Label(self.frame, text=doc_string).pack()
            if doc_string:
                doc_window = tk.Toplevel(self.window)
                doc_window.title("Documentation")
                doc_frame = ttk.Frame(doc_window)
                doc_frame.pack(padx=10, pady=10)
                doc_label = tk.Text(doc_frame, wrap='word', height=20, width=80)
                doc_label.insert('1.0', doc_string)
                doc_label.config(state='disabled')
                doc_label.pack()

        self.window.mainloop()

        # put plt back to normal
        plt.ioff()

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
        if default is None:
            default = 1
        slider = ttk.Scale(frame, from_=round(default*0.5, 2), to=round(default*1.5, 2), orient='horizontal')
        slider.set(default)
        slider.pack(side='right', fill='x', expand=True)

        value_label = ttk.Label(frame, text=str(default))
        value_label.pack(side='right')

        def update_value_label(event):
            value_label.config(text=f"{slider.get():.2f}")

        slider.bind("<Motion>", update_value_label)
        slider.bind("<ButtonRelease-1>", update_value_label)

        range_frame = ttk.Frame(self.frame)
        range_frame.pack(fill='x', padx=5, pady=5)

        min_label = ttk.Label(range_frame, text="Min")
        min_label.pack(side='left')

        min_textbox = ttk.Entry(range_frame, width=10)
        min_textbox.insert(0, default * 0.5)
        min_textbox.pack(side='left')

        max_label = ttk.Label(range_frame, text="Max")
        max_label.pack(side='left')

        max_textbox = ttk.Entry(range_frame, width=10)
        max_textbox.insert(0, default * 1.5)
        max_textbox.pack(side='left')

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
        default = self.defaults.get(param, 0)
        if default is None:
            default = ''
        textbox.insert(0, default)
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







