library(immuneSIM)
library(parallel)
library(pbmcapply)

insertion_and_deletion_lengths_df <- load_insdel_data()

starts <- 1:55
fx <- function(n)
{
  sim_repertoire_TRB <-immuneSIM(
    number_of_seqs = 1000,
    species = "hs",
    receptor = "tr",
    chain = "b",
    insertions_and_deletion_lengths = insertions_and_deletion_lengths_df,
    min_cdr3_length=7, 
    max_cdr3_length = 25,
    verbose= F)
  
  sim_repertoire_TRA <-immuneSIM(
    number_of_seqs = 1000,
    species = "hs",
    receptor = "tr",
    chain = "a",
    insertions_and_deletion_lengths = insertions_and_deletion_lengths_df,
    min_cdr3_length=7, 
    max_cdr3_length = 25,
    verbose= F)
  sim <- combine_into_paired(sim_repertoire_TRB, sim_repertoire_TRA)
  return(sim)
}


numCores <- detectCores() - 1
numCores
results <- pbmclapply(starts, fx, mc.cores = numCores)  

dt <- do.call("rbind", results)
rownames(dt) <- 1:nrow(dt)
write.csv(dt, '~/Documents/bertrand/data/immrep/simulated_cdr3ab_small.csv')

