"""Graphical User Interface (GUI) used for setting up the physical layout
of a computer display (ImageProjection class). To run GUI, use the following:
```python ImageProjectionGUI.py```

"""

import tkinter
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename

from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
import opencsp.common.lib.tool.tk_tools as tkt


class ImageProjectionGUI:
    """
    GUI for setting the physical layout parameters:
        'name'
        'win_size_x'
        'win_size_y'
        'win_position_x'
        'win_position_y'
        'size_x'
        'size_y'
        'position_x'
        'position_y'
        'projector_data_type'
        'projector_max_int'
        'shift_red_x'
        'shift_red_y'
        'shift_blue_x'
        'shift_blue_y'
    """

    def __init__(self):
        """Instantiates a new instance of the ImageProjection setup GUI in a new window"""
        # Define data names
        self.data_names = [
            'name',
            'win_size_x',
            'win_size_y',
            'win_position_x',
            'win_position_y',
            'size_x',
            'size_y',
            'position_x',
            'position_y',
            'projector_data_type',
            'projector_max_int',
            'image_delay',
            'shift_red_x',
            'shift_red_y',
            'shift_blue_x',
            'shift_blue_y',
            'ui_position_x',
        ]
        self.data_labels = [
            'Name',
            'Window X Size',
            'Window Y Size',
            'Window X Position',
            'Window Y Position',
            'Active Area X Size',
            'Active Area Y Size',
            'Active Area X Position',
            'Active Area Y Position',
            'Projector Image Data Type',
            'Projector Max Integer Value',
            'Image Delay (ms)',
            'Red Shift X',
            'Red Shift Y',
            'Blue Shift X',
            'Blue Shift Y',
            'GUI X Position',
        ]
        self.data_types = [str, int, int, int, int, int, int, int, int, str, int, int, int, int, int, int, int]

        # Declare variables
        self.projector: ImageProjection
        self.display_data: dict

        # Create tkinter object
        self.root = tkt.window()

        # Set title
        self.root.title('ImageProjection Setup')

        # Add all buttons/widgets to window
        self.create_layout()

        # Activate buttons
        self.activate_btns(False)

        # Load defaults and populate entries
        self.load_defaults()

        # Place window
        self.update_window_size()

        # Run window infinitely
        self.root.mainloop()

    def update_window_size(self):
        """Updates the window size to current set value"""
        # Set size and position of window
        self.root.geometry(f'500x650+{self.display_data["ui_position_x"]:d}+100')

    def create_layout(self):
        """Creates GUI widgets"""

        # Create data table
        self.data_cells = []
        for r, label in enumerate(self.data_labels):
            tkinter.Label(text=label, width=25).grid(row=r, column=0)
            e = tkinter.Entry(width=50)
            e.grid(row=r, column=1)
            self.data_cells.append(e)

        # Show projector button
        self.btn_show_proj = tkinter.Button(self.root, text='Show Display', command=self.show_projector)
        r += 1
        self.btn_show_proj.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')

        # Update projector button
        self.btn_update_proj = tkinter.Button(self.root, text='Update All', command=self.update_windows)
        r += 1
        self.btn_update_proj.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')

        # Close display button
        self.btn_close_proj = tkinter.Button(self.root, text='Close Display', command=self.close_projector)
        r += 1
        self.btn_close_proj.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')

        # Show crosshairs
        self.btn_crosshairs = tkinter.Button(self.root, text='Show Crosshairs', command=self.update_windows)
        r += 1
        self.btn_crosshairs.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')

        # Show axes button
        self.btn_axes = tkinter.Button(self.root, text='Show Display Axes', command=self.show_axes)
        r += 1
        self.btn_axes.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')

        # Show calibration image button
        self.btn_calib = tkinter.Button(self.root, text='Show calibration image', command=self.show_calibration_image)
        r += 1
        self.btn_calib.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')

        # Save as button
        self.btn_save = tkinter.Button(self.root, text='Save as HDF...', command=self.save_as)
        r += 1
        self.btn_save.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')

        # Load button
        self.btn_load = tkinter.Button(self.root, text='Load from HDF...', command=self.load_from)
        r += 1
        self.btn_load.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')

        # Close window button
        self.btn_close = tkinter.Button(self.root, text='Close All', command=self.close)
        r += 1
        self.btn_close.grid(row=r, column=0, pady=2, padx=2, sticky='nesw')

    def activate_btns(self, active: bool):
        """
        Activates/deactivtes buttons depending on if the projector screen is
        active.

        Parameters
        ----------
        active : bool
            If the projector screen is active.

        """
        if active:
            # States if projector is active
            active_projector = 'normal'
            inactive_projector = 'disabled'
        else:
            # States if projector is inactive
            active_projector = 'disabled'
            inactive_projector = 'normal'

        # Enable/disable buttons
        self.btn_show_proj['state'] = inactive_projector
        self.btn_update_proj['state'] = active_projector
        self.btn_close_proj['state'] = active_projector
        self.btn_axes['state'] = active_projector
        self.btn_crosshairs['state'] = active_projector
        self.btn_calib['state'] = active_projector

    def show_projector(self):
        """
        Opens the projector window.

        """
        # Get user data
        self.get_user_data()

        # Update GUI size
        self.update_window_size()

        # Create a new Toplevel window
        projector_root = tkt.window(self.root, Toplevel=True)
        self.projector = ImageProjection(projector_root, self.display_data)

        # Activate buttons
        self.activate_btns(True)

    def update_windows(self):
        """
        Updates the projector active area data, updates the window size, and
        shows crosshairs.

        """
        # Read data from entry cells
        self.get_user_data()

        # Update size of window
        self.projector.upate_window(self.display_data)

        # Show crosshairs
        self.projector.show_crosshairs()

        # Update position of UI window
        self.update_window_size()

    def show_axes(self):
        """Shows axis labels."""
        # Update active area of projector
        self.update_windows()

        # Show X/Y axes
        self.projector.show_axes()

    def show_calibration_image(self):
        """Shows calibration image"""
        # Update active area of projector
        self.update_windows()

        # Show cal image
        self.projector.show_calibration_image()

    def close_projector(self):
        """
        Closes projector window.

        """
        # Close projector
        self.projector.close()

        # Disable buttons
        self.activate_btns(False)

    def save_as(self):
        """
        Saves physical layout parameters in HDF file

        """
        # Get save file name
        file = asksaveasfilename(defaultextension='.h5', filetypes=[("HDF5 File", "*.h5")])

        # Save file as HDF
        if file != '':
            self.save_to_hdf(file)

    def load_from(self):
        """
        Loads physical layout parameters from HDF file

        """
        # Get file name
        file = askopenfilename(defaultextension='.h5', filetypes=[("HDF5 File", "*.h5")])

        # Load file
        if file != '':
            self.load_from_hdf(file)

    def get_user_data(self):
        """
        Reads user input data and saves in dictionary.

        """
        # Check inputs format
        if not self.check_inputs():
            return

        # Gets data from user input boxes and saves in class
        data = {}
        for dtype, name, entry in zip(self.data_types, self.data_names, self.data_cells):
            data.update({name: dtype(entry.get())})
        self.display_data = data

    def check_inputs(self) -> bool:
        """
        Checks that user inputs are formatted correctly

        Returns
        -------
        bool
            True if all inputs are formatted correctly

        """
        # Checks inputs are correct
        for name, dtype, entry in zip(self.data_labels, self.data_types, self.data_cells):
            try:
                dtype(entry.get())
            except ValueError:
                messagebox.showerror('Invalid input', f'Input for "{name:s}" must be {dtype}')
                return False
        return True

    def set_user_data(self):
        """
        Sets the loaded user data in the user input boxes

        """
        for name, entry in zip(self.data_names, self.data_cells):
            entry.delete(0, tkinter.END)
            entry.insert(0, self.display_data[name])

    def load_defaults(self):
        """
        Sets default values.

        """
        self.display_data = {
            'name': 'Default Image Projection',
            'win_size_x': 800,
            'win_size_y': 500,
            'win_position_x': 0,
            'win_position_y': 0,
            'size_x': 700,
            'size_y': 400,
            'position_x': 50,
            'position_y': 50,
            'projector_data_type': 'uint8',
            'projector_max_int': 255,
            'image_delay': 400,
            'shift_red_x': 0,
            'shift_red_y': 0,
            'shift_blue_x': 0,
            'shift_blue_y': 0,
            'ui_position_x': 100,
        }
        self.set_user_data()

    def load_from_hdf(self, file: str):
        """
        Loads saved user data from HDF file

        Parameters
        ----------
        file : str
            File to load.

        """
        # Load data
        self.display_data = ImageProjection.load_from_hdf(file)

        # Set data in input fields
        self.set_user_data()

    def save_to_hdf(self, file: str):
        """
        Saves user data to HDF file.

        Parameters
        ----------
        file : str
            File to save to.

        """
        # Load user data
        self.get_user_data()

        # Save as HDF file
        ImageProjection.save_to_hdf(self.display_data, file)

    def close(self):
        """
        Closes all windows

        """
        self.root.destroy()


if __name__ == '__main__':
    ImageProjectionGUI()
