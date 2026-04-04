from ansys.dpf import core as dpf
import pandas as pd
import numpy as np

def rst_to_parquet(rst_path, csv_path):
    server = dpf.start_local_server(config=dpf.AvailableServerConfigs.InsecureGrpcServer)

    model = dpf.Model(data_sources=rst_path)

    strain_op = dpf.operators.result.elastic_strain()
    strain_op.inputs.data_sources.connect(model.metadata.data_sources)
    strain_op.inputs.requested_location.connect(dpf.locations.elemental)
    strain_op.inputs.time_scoping.connect([model.metadata.time_freq_support.n_sets])
    strain_op.inputs.mesh_scoping.connect(model.metadata.meshed_region.elements.scoping)
    strain = strain_op.outputs.fields_container()[0]

    disp_op = dpf.operators.result.displacement()
    disp_op.inputs.data_sources.connect(model.metadata.data_sources)
    disp_op.inputs.time_scoping.connect([model.metadata.time_freq_support.n_sets])
    disp_op.inputs.mesh_scoping.connect(model.metadata.meshed_region.elements.scoping)
    disp_elemental = disp_op.outputs.fields_container()[0]

    centroids_op = dpf.operators.result.element_centroids()
    centroids_op.inputs.data_sources.connect(model.metadata.data_sources)
    centroids_op.inputs.mesh.connect(model.metadata.meshed_region)
    centroids = (centroids_op.outputs.fields_container())[0]

    df = pd.DataFrame(
        strain.data[1::3],
        index=centroids.scoping.ids,
        columns=["EXX", "EYY", "EZZ", "EXY", "EYZ", "EXZ"],
    )

    df.insert(0, "X", pd.Series(np.round(centroids.data[:, 0], 3), index=centroids.scoping.ids))
    df.insert(1, "Y", pd.Series(np.round(centroids.data[:, 1], 3), index=centroids.scoping.ids))
    df.insert(2, "Z", pd.Series(np.round(disp_elemental.data[:, 2], 3), index=centroids.scoping.ids))

    df.index.name = "Element ID"

    df.to_parquet(csv_path)

def extract_shell_layer_to_parquet(rst_path, csv_output_path, layer_type="Top"):
    """
    Extracts results for a specific shell layer (e.g., Top, Bottom, Mid) from an RST file
    and saves them to a Parquet file, including elemental Z displacement.

    Args:
        rst_path (str or Path): Path to the RST file.
        csv_output_path (str or Path): Path to save the output Parquet file.
        layer_type (str): The type of shell layer to extract.
                          Supported values: "Top", "Bottom", "Mid".
    """
    server = dpf.start_local_server(config=dpf.AvailableServerConfigs.InsecureGrpcServer)
    model = dpf.Model(data_sources=rst_path)

    layer_enum_map = {
        "Top": dpf.shell_layers.top,
        "Bottom": dpf.shell_layers.bottom,
        "Mid": dpf.shell_layers.mid,
        "TopBottom": dpf.shell_layers.topbottom,
        "TopBottomMid": dpf.shell_layers.topbottommid
    }

    if layer_type not in layer_enum_map:
        raise ValueError(f"Unsupported layer_type: {layer_type}. Choose from {list(layer_enum_map.keys())}")

    e_shell_layer_val = layer_enum_map[layer_type]

    # 1. Get Elastic Strain results (elemental_nodal)
    strain_op = dpf.operators.result.elastic_strain()
    strain_op.inputs.data_sources.connect(model.metadata.data_sources)
    strain_op.inputs.requested_location.connect(dpf.locations.elemental)
    strain_op.inputs.time_scoping.connect([model.metadata.time_freq_support.n_sets])
    strain_fc = strain_op.outputs.fields_container()

    # Apply change_shell_layers operator to strain
    change_layer_strain_op = dpf.operators.utility.change_shell_layers()
    change_layer_strain_op.inputs.fields_container.connect(strain_fc)
    change_layer_strain_op.inputs.e_shell_layer.connect(e_shell_layer_val)
    change_layer_strain_op.inputs.mesh.connect(model.metadata.meshed_region)
    layer_strain_field = change_layer_strain_op.outputs.fields_container_as_fields_container()[0]

    # 2. Get Displacement results (Nodal by default, converting to Elemental)
    disp_op = dpf.operators.result.displacement()
    disp_op.inputs.data_sources.connect(model.metadata.data_sources)
    disp_op.inputs.time_scoping.connect([model.metadata.time_freq_support.n_sets])
    
    # Average nodal displacement to elemental displacement
    nodal_to_elem_op = dpf.operators.averaging.nodal_to_elemental_fc()
    nodal_to_elem_op.inputs.fields_container.connect(disp_op.outputs.fields_container())
    nodal_to_elem_op.inputs.mesh.connect(model.metadata.meshed_region)
    disp_elem_fc = nodal_to_elem_op.outputs.fields_container()

    # 3. Get Centroids for the elements
    centroids_op = dpf.operators.result.element_centroids()
    centroids_op.inputs.data_sources.connect(model.metadata.data_sources)
    centroids_op.inputs.mesh.connect(model.metadata.meshed_region)
    layer_centroids_field = centroids_op.outputs.fields_container()[0]

    # 4. Rescope elemental displacement to match the specific layer's elements
    rescope_op = dpf.operators.scoping.rescope()
    rescope_op.inputs.fields.connect(disp_elem_fc)
    rescope_op.inputs.mesh_scoping.connect(layer_centroids_field.scoping)
    rescope_op.inputs.default_value.connect(0.0)
    disp_elem_field_rescoped = rescope_op.outputs.fields_as_fields_container()[0]

    # 5. Format Data for Pandas
    strain_data_reshaped = layer_strain_field.data.reshape(-1, layer_strain_field.component_count)
    elem_ids = layer_centroids_field.scoping.ids

    df = pd.DataFrame(
        strain_data_reshaped,
        index=elem_ids,
        columns=["EXX", "EYY", "EZZ", "EXY", "EYZ", "EXZ"],
    )

    # Note: .data must be used to extract the NumPy array from the PyDPF Field before slicing
    df.insert(0, "X", pd.Series(np.round(layer_centroids_field.data[:, 0], 3), index=elem_ids))
    df.insert(1, "Y", pd.Series(np.round(layer_centroids_field.data[:, 1], 3), index=elem_ids))
    
    # Insert Elemental Z Displacement (Component index 2 is Z)
    df.insert(2, "Z", pd.Series(disp_elem_field_rescoped.data[:, 2], index=elem_ids))

    df.index.name = "Element ID"
    df.to_parquet(csv_output_path)