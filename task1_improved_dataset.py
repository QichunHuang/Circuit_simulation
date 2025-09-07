import numpy as np
import polars as pl

from coupled_resonator_tools import calculate_impedance_spectrum


def generate_coupling_matrix_matlab_style(N, kk):
    """Generate coupling matrix using MATLAB style"""
    kthresh_type = 1
    kthresh_idx = 5

    j_indices, jj_indices = np.meshgrid(range(N), range(N), indexing="ij")
    distance = np.abs(j_indices - jj_indices)

    K = np.where(distance == 0, 0, ((-1) ** np.sign(distance)) * (kk**distance))
    if kthresh_type == 1:
        K = np.where(distance > kthresh_idx, 0, K)

    return K


def generate_L_variations(L_base, L_num, variation_index, variation_percent):
    """
    Generate L variations focused on specific index (corrected version)

    Args:
        L_base: Base inductance array
        L_num: Number of variations to generate
        variation_index: Index to vary (0-based, should be 3 for L_fixed[3])
        variation_percent: Variation percentage (0.1 for ±10%)

    Returns:
        Array of L variations with shape (L_num, len(L_base))
    """
    L_base = np.array(L_base, dtype=float)
    l_k = L_base[variation_index]
    l_k_min = l_k * (1 - variation_percent)
    l_k_max = l_k * (1 + variation_percent)

    # Generate systematic sampling: 500 evenly spaced points
    values = np.linspace(l_k_min, l_k_max, L_num)

    # Create array with L_base repeated L_num times
    ret = np.tile(L_base, (L_num, 1))
    # Vary only the specified index
    ret[:, variation_index] = values

    return ret


def get_frequency_range(L_variations, C, F_num):
    """Calculate frequency range for impedance spectrum"""
    import math

    min_fn_val = float("inf")
    max_fn_val = float("-inf")

    for L in L_variations:
        # Calculate resonant frequencies: f = 1/(2π√LC)
        fn_val = (1.0 / (2.0 * math.pi)) * np.sqrt(1.0 / (L * C))
        min_fn_val = min(np.min(fn_val), min_fn_val)
        max_fn_val = max(np.max(fn_val), max_fn_val)

    # Extend frequency range
    f_start = min_fn_val * 0.8
    f_end = max_fn_val * 1.5
    fax = np.linspace(f_start, f_end, F_num)

    return fax


def process_impedance_spectrum(frequency, impedance_spectrum):
    """
    Process the full impedance spectrum for dataset generation

    Args:
        frequency: Frequency array (F_num points)
        impedance_spectrum: Complex impedance spectrum (F_num points)

    Returns:
        z_magnitudes: Impedance magnitudes (F_num points)
        z_reals: Real parts of impedance (F_num points)
        z_imags: Imaginary parts of impedance (F_num points)
    """
    z_magnitudes = np.abs(impedance_spectrum)
    z_reals = np.real(impedance_spectrum)
    z_imags = np.imag(impedance_spectrum)

    return z_magnitudes, z_reals, z_imags


def generate_datasets(L_index):
    """Generate both z-magnitude and complex z-value datasets
    
    Args:
        L_index: Index of inductance to vary (0-4 for L1-L5)
        
    Returns:
        tuple: (z_df, mixed_z_df) - z-magnitude and complex impedance DataFrames
    """

    # Circuit parameters (matching specification requirements)
    N = 5                    # Number of resonators/coils
    F_num = 1500            # Number of frequency points
    L_num = 1000            # Number of training samples per variation
    coil_driven = 3         # Driven coil index (1-based)
    variation_percent = 0.1  # Inductance variation range (±10%)

    # Base circuit configuration from specification
    L_base = np.array([12, 12, 12, 12, 12]) * 1e-6      # Base inductances [H]
    L_fixed = L_base + np.array([0, 0, 1, 1, 1]) * 1e-6  # Modified inductances [H]

    # Fixed circuit parameters
    C_fixed = np.array([162, 184, 150, 210, 169]) * 1e-12    # Base capacitances [F]
    R_fixed = np.array([25, 25, 50, 25, 25])                  # Resistances [Ohms]
    C_fixed = C_fixed + np.array([0, 0, 20, 20, 20]) * 1e-12  # Additional capacitances [F]
    kk = 0.15  # Maximum coupling coefficient

    print("Generating dataset with following configuration:")
    print(f"L_base: {L_base * 1e6} μH")
    print(f"L_fixed: {L_fixed * 1e6} μH")
    print(
        f"Varying L_fixed[{L_index}] = {L_fixed[L_index] * 1e6:.1f} μH by ±{variation_percent * 100}%"
    )
    print(
        f"Variation range: {L_fixed[L_index] * (1 - variation_percent) * 1e6:.1f} - {L_fixed[L_index] * (1 + variation_percent) * 1e6:.1f} μH"
    )

    # Generate inductance variations for specified index
    L_variations = generate_L_variations(L_fixed, L_num, L_index, variation_percent)

    # Generate coupling matrix using MATLAB-style approach
    K_matrix = generate_coupling_matrix_matlab_style(N, kk)
    print(f"Coupling matrix shape: {K_matrix.shape}")

    # Calculate optimal frequency range based on resonant frequencies
    frequency_axis = get_frequency_range(L_variations, C_fixed, F_num)
    print(
        f"Frequency range: {frequency_axis[0] / 1e6:.2f} - {frequency_axis[-1] / 1e6:.2f} MHz"
    )

    # Initialize data storage lists
    z_data = []       # Store impedance magnitude data
    mixed_z_data = [] # Store complex impedance data (real + imaginary)

    print(f"Processing {L_num} inductance variations...")

    # Process each inductance variation
    for i, L_current in enumerate(L_variations):
        if i % 50 == 0:
            print(f"Processing sample {i + 1}/{L_num}")

        # Set up circuit parameters for simulation
        params = {"N": N, "L": L_current, "C": C_fixed, "R": R_fixed, "K": K_matrix}

        # Calculate complex impedance spectrum across frequency range
        impedance_spectrum = calculate_impedance_spectrum(
            frequency_axis, params, coil_driven
        )

        # Extract magnitude, real, and imaginary components
        z_magnitudes, z_reals, z_imags = process_impedance_spectrum(
            frequency_axis, impedance_spectrum
        )

        # Convert inductance values to microhenries for output
        L_uH = L_current * 1e6

        # Prepare z-magnitude dataset row: [mag_1, mag_2, ..., mag_F_num, L1, L2, L3, L4, L5]
        z_row = list(z_magnitudes) + list(L_uH)
        z_data.append(z_row)

        # Prepare complex z dataset row: [real_1, imag_1, real_2, imag_2, ..., L1, L2, L3, L4, L5]
        mixed_z_row = []
        for j in range(F_num):
            mixed_z_row.extend([z_reals[j], z_imags[j]])
        mixed_z_row.extend(L_uH)
        mixed_z_data.append(mixed_z_row)

    # Create descriptive column names for datasets
    # Complex impedance columns: z_real_0, z_imag_0, z_real_1, z_imag_1, ..., L1, L2, L3, L4, L5
    mixed_z_columns = []
    for i in range(F_num):
        mixed_z_columns.extend([f'z_real_{i}', f'z_imag_{i}'])
    mixed_z_columns.extend([f'L{i+1}' for i in range(N)])
    
    # Magnitude impedance columns: z_mag_0, z_mag_1, ..., z_mag_1499, L1, L2, L3, L4, L5
    z_columns = [f'z_mag_{i}' for i in range(F_num)] + [f'L{i+1}' for i in range(N)]
    
    # Create DataFrames using Polars with explicit row orientation
    z_df = pl.DataFrame(z_data, schema=z_columns, orient="row")
    mixed_z_df = pl.DataFrame(mixed_z_data, schema=mixed_z_columns, orient="row")
    
    # Save datasets
    z_data_file = f"z_dataset_{L_index}.csv"
    mixed_data_file = f"mixed_z_dataset_{L_index}.csv"
    
    z_df.write_csv(z_data_file)
    mixed_z_df.write_csv(mixed_data_file)
    print(
        f"Saved {z_data_file} with shape: {z_df.shape} (should be {L_num}×{F_num + N})"
    )
    print(
        f"Saved {mixed_data_file} with shape: {mixed_z_df.shape} (should be {L_num}×{2 * F_num + N})"
    )

    # Display sample data using Polars syntax
    print("\nSample from z_dataset.csv (first 3 rows, first 5 z-columns and L-columns):")
    sample_cols = [f'z_mag_{i}' for i in range(5)] + [f'L{i+1}' for i in range(N)]
    print(z_df.select(sample_cols).head(3))

    print("\nSample from mixed_z_dataset.csv (first 3 rows, first 10 columns):")
    print(mixed_z_df.select(mixed_z_df.columns[:10]).head(3))

    # Basic statistics
    print("\nDataset statistics:")
    print(
        f"L_fixed[{L_index}] variation range: {L_variations[:, L_index].min() * 1e6:.2f} - {L_variations[:, L_index].max() * 1e6:.2f} μH"
    )
    
    # Get Z magnitude statistics using Polars
    z_mag_cols = [f'z_mag_{i}' for i in range(F_num)]
    z_min = z_df.select(z_mag_cols).min().min_horizontal().item()
    z_max = z_df.select(z_mag_cols).max().max_horizontal().item()
    
    print(f"Z magnitude range: {z_min:.2f} - {z_max:.2f} Ohms")
    print(
        f"Expected dimensions: z_dataset.csv = {L_num}×{F_num + N}, mixed_z_dataset.csv = {L_num}×{2 * F_num + N}"
    )

    return z_df, mixed_z_df


def main():
    """Main function to generate datasets for all L variations (L1-L5)"""
    try:
        print("Starting dataset generation for all L variations...")
        
        for i in range(0, 5):  # Fixed: should be 0-5 for L1-L5
            print(f"\n{'='*50}")
            print(f"Generating datasets for L{i+1} variation...")
            print(f"{'='*50}")
            
            z_dataset, mixed_z_dataset = generate_datasets(i)
            
            print(f"\n✅ L{i+1} dataset generation completed successfully!")
            print("Generated files:")
            print(f"  - z_dataset_{i}.csv (z-magnitude values)")
            print(f"  - mixed_z_dataset_{i}.csv (complex z-values)")
        
        print("\n" + "="*60)
        print("🎉 All dataset generation completed successfully!")
        print("Generated 10 files total (5 z-magnitude + 5 complex z-datasets)")
        print("="*60)

    except Exception as e:
        print(f"❌ Error during dataset generation: {e}")
        raise


if __name__ == "__main__":
    main()
