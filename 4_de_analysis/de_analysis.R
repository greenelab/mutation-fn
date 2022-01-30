# functions to perform RNA-seq differential expression analysis using DESeq2
# mostly copied from and inspired by:
# https://github.com/greenelab/generic-expression-patterns/blob/master/generic_expression_patterns_modules/DE_analysis.R

library("DESeq2")

get_DE_stats_DESeq <- function(metadata_file,
                               expression_file,
                               experiment_id,
                               output_dir) {

  # This function performs DE analysis using DESeq.
  # Expression data in expression_file are grouped based on metadata_file
  #
  # Arguments
  # ---------
  # metadata_file: str
  #   File containing mapping between sample id and group
  #
  # expression_file: str
  #   File containing gene expression data
  #
  # experiment_id: str
  #   Experiment id used to label saved output filee
  #
  # output_dir: str
  #   Directory to save output files to

  expression_data <- t(as.matrix(read.csv(expression_file, sep="\t", header=TRUE, row.names=1)))
  metadata <- as.matrix(read.csv(metadata_file, sep="\t", header=TRUE, row.names=1))

  print("Checking sample ordering...")
  print(all.equal(colnames(expression_data), rownames(metadata)))

  group <- interaction(metadata[,1])
  mm <- model.matrix(~0 + group)

  # Note about DESeq object
  # The dataset we have is a matrix of estimated counts that were combined across samples. We don't have the individual raw data files output from RSEM,
  # which is needed to call tximport.
  # See here for rationale behind using DESeqDataSetFromMatrix rather than tximport
  # (should apply here as well):
  # https://github.com/greenelab/generic-expression-patterns/blob/f2f7488217bdd197d2a5ed6c2512f729b91a1a45/generic_expression_patterns_modules/DE_analysis.R#L134
  ddset <- DESeqDataSetFromMatrix(expression_data, colData=metadata, design = ~group)
  deseq_object <- DESeq(ddset, quiet=TRUE)

  # Note parameter settings:
  # `independentFilter=False`: We have turned off the automatic filtering, which
  # filter out those tests from the procedure that have no or little
  # chance of showing significant evidence, without even looking at their test statistic.
  # Typically, this results in increased detection power at the same experiment-wide
  # type I error, as measured in terms of the false discovery rate.
  # cooksCutoff=True (default): Cook's distance as a diagnostic to tell if a single sample
  # has a count which has a disproportionate impact on the log fold change and p-values.
  # These genes are flagged with an NA in the pvalue and padj columns
  deseq_results <- results(deseq_object, independentFiltering=FALSE)
  deseq_results_df <-  as.data.frame(deseq_results)

  # Save summary statistics of DEGs
  out_file = paste(output_dir, "/DE_stats_", experiment_id, ".txt", sep="")
  write.table(deseq_results_df, file = out_file, row.names = T, sep = "\t", quote = F)
}
