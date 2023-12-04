import numpy as np
import matplotlib.pyplot as plt


class MultiResolutionPDF:
    """
    A class for managing and visualizing probability density functions (PDFs)
    in a multi-resolution format.

    This class allows for adding data in the form of bins, normalizing the bins, 
    computing statistical properties (mean, mode, and standard deviation), plotting 
    the PDF, and evaluating the PDF at a given point.

    Attributes:
        bin_center_arr (numpy.array): Stores the centers of the bins.
        bin_width_arr (numpy.array): Stores the widths of the bins.
        bin_height_arr (numpy.array): Stores the heights of the bins.
        mode (float): The mode of the PDF, computed in `compute_stats`.
        mean (float): The mean of the PDF, computed in `compute_stats`.
        sigma (float): The standard deviation of the PDF, computed in `compute_stats`.
    """
    def __init__(self):
        """
        Constructor for the MultiResolutionPDF class.

        Initializes arrays for bin centers, widths, and heights. Statistical properties
        (mode, mean, sigma) are initialized to None.
        """
        self.bin_center_arr = np.array([])
        self.bin_width_arr = np.array([])
        self.bin_height_arr = np.array([])
        self.mode = None
        self.mean = None
        self.sigma = None

    def add_bin(self, center_arr, width_arr, height_arr):
        """
        Adds bins to the PDF.

        Args:
            center_arr (array_like): Array or list of bin centers.
            width_arr (array_like): Array or list of bin widths.
            height_arr (array_like): Array or list of bin heights.

        Raises:
            AssertionError: If the lengths of center_arr, width_arr, and height_arr are not equal.
        """
        assert len(center_arr) == len(width_arr) == len(height_arr), "center_arr, width_arr, height_arr must have the same length"
        self.bin_center_arr = np.append(self.bin_center_arr, center_arr)
        self.bin_width_arr = np.append(self.bin_width_arr, width_arr)
        self.bin_height_arr = np.append(self.bin_height_arr, height_arr)

    def normalize(self, report = False):
        """
        Normalizes the PDF so that the total area under the bins equals 1.
        Prints the total area before and after normalization.
        """
        total_area = np.sum(self.bin_width_arr * self.bin_height_arr)
        if report:
            if total_area == 1.:
                print('already normalized')
            else:
                print('total area before normalization:', total_area)
                self.bin_height_arr = self.bin_height_arr / total_area
            
    def compute_stats(self):  
        """
        Computes and updates the statistical properties of the PDF: mean, mode, and standard deviation (sigma).
        """
        self.mean = np.sum(self.bin_center_arr * self.bin_width_arr * self.bin_height_arr)
        self.mode = self.bin_center_arr[np.argmax(self.bin_height_arr)]
        variance = np.sum(
            (self.bin_center_arr-self.mean) ** 2 * self.bin_height_arr * self.bin_width_arr
        )
        self.sigma = np.sqrt(variance)
            
    def plot(self, ax=None, log_scale=False):
        """
        Plots the PDF as a bar chart.

        Args:
            ax (matplotlib.axes.Axes, optional): Matplotlib axis object. If None, a new figure and axis are created.
            log_scale (bool, optional): If True, sets the y-axis to logarithmic scale.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 4), dpi=100)

        ax.bar(self.bin_center_arr, self.bin_height_arr, width=self.bin_width_arr, align='center', color='black', alpha=0.5)
        ax.vlines(self.mean, 0, np.max(self.bin_height_arr), color='blue', label='Mean', lw=2)
        ax.vlines(self.mode, 0, np.max(self.bin_height_arr), color='lightblue', label='Mode', lw=2)
        
        # Visualize sigma as horizontal lines
        ax.hlines(y=np.max(self.bin_height_arr), xmin=self.mean - self.sigma, xmax=self.mean + self.sigma, color='g', label='Sigma', lw=2)

        if log_scale:
            ax.set_yscale('log')

        ax.legend()

        # If ax was None, show the plot
        if ax is None:
            plt.show()
        
    def check_overlap(self):
        ### Check that bins do not overlap
        NotImplemented

    def value_at(self, x):
        for center, width, height in zip(self.bin_center_arr, self.bin_width_arr, self.bin_height_arr):
            if center - width / 2 <= x <= center + width / 2:
                return height
        return 0

# # Example usage
# pdf = MultiResolutionPDF()
# # pdf.add_bin(center_arr =np.array([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]), 
# #             width_arr =np.array([1]*10), 
# #             height_arr =np.array([0.1]*10)
# #             # height_arr =np.array([0,1,2,3,4,5,6,7,8,9])
# #             )

# pdf.add_bin(center_arr =np.array([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,9.5]), 
#             width_arr =np.array([1]*9), 
#             height_arr =np.array([0.00061674,0.00689466,0.00428099,0.00240142,0.00349404,0.00377796,0.00668253,0.01719828,0.01565921])
#             )
# pdf.add_bin(center_arr =np.array([8.05,8.15,8.25,8.35,8.45,8.55,8.65,8.75,8.85,8.95]), 
#             width_arr =np.array([0.1]*10), 
#             height_arr =np.array([0.03162489,0.03137879,0.04530057,0.083974,0.11750073,0.27749962,0.99931455,1.476891,5.4022436,0.92421484])
#             )
# pdf.normalize()
# pdf.compute_stats()

# pdf.plot(log_scale=True)