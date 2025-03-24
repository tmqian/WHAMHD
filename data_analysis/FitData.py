import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as opt
from scipy.optimize import least_squares


# Perform forward projection by integrating radial model along chords
def forward_integrate(radial_model, chord_positions, integration_points, **model_params):
    """
    Computes line integrals of a radial model along parallel chords.
    
    Parameters:
    - radial_model (function): The radial model function to integrate.
    - chord_positions (array-like): Y-coordinates of chord impact parameters.
    - integration_points (array-like): X-coordinates for numerical integration.
    - model_params (dict): Additional parameters for the radial model.
    
    Returns:
    - array-like: Line integrals of the radial model along each chord.
    """
    line_integrals = []
    for y in chord_positions:
        # Calculate radial positions along the chord using Pythagorean theorem
        radial_positions = np.sign(y) * np.sqrt(integration_points**2 + y**2)
        # Evaluate model at each radial position
        model_values = radial_model(radial_positions, **model_params)
        # Perform numerical integration using rectangle method
        step_size = np.diff(integration_points)[0]
        line_integrals.append(np.sum(model_values) * step_size)
    return np.array(line_integrals)

# Define objective function for optimization
def objective_function(params, radial_model, chord_positions, integration_points, 
                      measured_integrals, param_names):
    """
    Calculates sum of squared errors between model predictions and measurements.
    
    Parameters:
    - params (array-like): Parameter values to be optimized.
    - radial_model (function): The radial model function to optimize.
    - chord_positions (array-like): Y-coordinates of chord impact parameters.
    - integration_points (array-like): X-coordinates for integration.
    - measured_integrals (array-like): Measured line integral values.
    - param_names (list): Names of parameters in the radial model.
    
    Returns:
    - float: Sum of squared errors between model and measurements.
    """
    # Convert flat parameter array to dictionary for model function
    param_dict = dict(zip(param_names, params))
    # Calculate predicted integrals using current parameters
    predicted_integrals = forward_integrate(radial_model, chord_positions, 
                                         integration_points, **param_dict)
    # Return sum of squared errors
    return np.sum((predicted_integrals - measured_integrals) ** 2)


# Optimize model parameters to fit measured data
def optimize_radial_model(chord_positions, measured_integrals, radial_model, 
                        integration_points=np.linspace(-30, 30, 100),
                        param_names=None, 
                        initial_guess=None, 
                        optimization_method=None):
    """
    Finds optimal parameters for a radial model to match measured line integrals.
    
    Parameters:
    - chord_positions (array-like): Y-coordinates of chord impact parameters.
    - measured_integrals (array-like): Measured line integral values.
    - radial_model (function): The radial model function to optimize.
    - integration_points (array-like, optional): X-coordinates for integration.
    - param_names (list, optional): Names of parameters in the radial model.
    - initial_params (array-like, optional): Initial guess for parameter values.
    - optimization_method (str, optional): Optimization method for scipy.optimize.minimize.
    
    Returns:
    - dict: Optimized parameter values.
    """
    # Perform optimization using scipy
    result = opt.minimize(
        objective_function, 
        initial_guess, 
        args=(radial_model, chord_positions, integration_points, 
              measured_integrals, param_names),
        method=optimization_method,
    )
    
    # Convert optimized parameters to dictionary
    optimized_params = dict(zip(param_names, result.x))
    print("Optimized parameters:", optimized_params)
    
    # Plot results
    visualize_results(measured_integrals, chord_positions, radial_model, optimized_params, 
                     integration_points)
    
    return optimized_params


def visualize_results(measured_integrals, chord_positions, radial_model, optimized_params, 
                     integration_points=np.linspace(-30, 30, 100)):
    """
    Visualizes optimization results with comparison to measurements.
    
    Parameters:
    - measured_integrals (array-like): Measured line integral values.
    - chord_positions (array-like): Y-coordinates of chord impact parameters.
    - radial_model (function): The optimized radial model function.
    - optimized_params (dict): Optimized parameter values.
    - integration_points (array-like, optional): X-coordinates for integration.
    """
    # Generate radial positions for plotting the model function
    radial_positions = np.linspace(-20, 20, 50)
    
    # Create figure with two panels
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Left Panel: Forward Model vs. Data
    predicted_integrals = forward_integrate(radial_model, chord_positions, 
                                          integration_points, **optimized_params)
    axs[0].scatter(chord_positions, measured_integrals, label="Measured Data", color="b")
    axs[0].plot(chord_positions, predicted_integrals, label="Model Prediction", 
               linestyle="--", color="r")
    axs[0].set_xlabel("Chord Impact Parameter (y)")
    axs[0].set_ylabel("Line Integral")
    axs[0].set_title("Measured Data vs. Forward Model")
    
    # Right Panel: Optimized Radial Profile
    axs[1].plot(radial_positions, radial_model(radial_positions, **optimized_params), 
               label="Optimized Profile", color="orange")
    axs[1].set_xlabel("Radius (r)")
    axs[1].set_ylabel("Model Value f(r)")
    axs[1].set_title("Reconstructed Radial Profile")
    
    # Format both panels
    for ax in axs:
        ax.legend(loc=1)
        ax.grid()
        ax.set_ylim(bottom=0)
    
    # Show plots
    plt.tight_layout()
    plt.show()
