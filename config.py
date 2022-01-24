import pathlib

repo_root = pathlib.Path(__file__).resolve().parent

# important subdirectories
data_dir = repo_root / 'data'

# differential expression directories/files
de_data_dir = repo_root / '4_de_analysis' / 'data'
raw_de_data_dir = de_data_dir / 'raw'
processed_counts_file = de_data_dir / 'tcga_processed_counts.tsv.gz'
de_sample_info = de_data_dir / 'tcga_expression_sample_identifiers.tsv'

# location of mpmp repo, some analyses depend on this
mpmp_location = pathlib.Path('~/research/mpmp/')

# location of vogelstein genes
vogelstein_base_url = "https://github.com/greenelab/pancancer/raw"
vogelstein_commit = "2a0683b68017fb226f4053e63415e4356191734f"
