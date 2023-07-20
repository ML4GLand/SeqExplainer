import os
import tempfile
import pandas as pd

def annotate_pfms(
	filename,
	motifs_db, 
    output_dir=None,
    out_filename=None,
	tomtom_exec_path='tomtom',
	is_writing_tomtom_matrix: bool = True
):
	if output_dir is None:
		output_dir = os.getcwd()

	_, tomtom_fname = tempfile.mkstemp()
	out_filename = "motifs" if out_filename is None else out_filename

	# run tomtom
	cmd = '%s -no-ssc -oc . --verbosity 1 -text -min-overlap 5 -dist pearson -evalue -thresh 10.0 %s %s > %s' % (tomtom_exec_path, filename, motifs_db, tomtom_fname)
	os.system(cmd)
	tomtom_results = pd.read_csv(tomtom_fname, sep="\t")

	if is_writing_tomtom_matrix:
		output_subdir = os.path.join(output_dir, "tomtom")
		os.makedirs(output_subdir, exist_ok=True)
		output_filepath = os.path.join(output_subdir, out_filename)
		os.system(f'mv {tomtom_fname} {output_filepath}')
	else:
		os.system('rm ' + tomtom_fname)
	return tomtom_results