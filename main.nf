#!/usr/bin/env nextflow

// Define parameters
params.py_script = "/home/users/allstaff/whitehead/nexflow_halo_stitch/Stitch_HALO_final.py" // The path to your Python script
params.input_path = "" // The path to your input data
params.help = false

// Define help message
helpMessage = """
Nextflow pipeline to run stitching of halo files"
Usage: nextflow run my_pipeline.nf --input_path <Path to .tif tiles>


Options:
  --input_path    Path to the input path.
  --help          Display this help message.
"""


if (params.help) {
    println helpMessage
    exit 0
}

process run_script {
    container '/stornext/Img/data/prkfs1/m/Microscopy/BAC_Conda_envs/nextlfow_containers/nf_stitch.sif'

    //Resource Requirements
    cpus 12
    memory '512 GB'
    time '2h'

    input:
    path(py_script)
    path(input_path)

    script:
    """
    . /opt/conda/etc/profile.d/conda.sh
    conda activate nf_stitch
    python ${py_script} ${input_path}
    """
}

workflow {
    run_script(file(params.py_script), file(params.input_path) )
}
